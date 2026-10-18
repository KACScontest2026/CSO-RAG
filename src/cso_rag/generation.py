
import hashlib,json,os,subprocess,threading,time
from pathlib import Path
from .core import read,save,digest,aggregate
from .validation import strict_object,repeat_reason,validate

def load_model(cfg,vocab_only=False):

    global DLL_HANDLES
    dll_dirs=[str(Path(p).resolve()) for p in cfg.get('runtime_dll_dirs',[])] if os.name=='nt' else []
    DLL_HANDLES=[os.add_dll_directory(p) for p in dll_dirs]


    if dll_dirs:os.environ['PATH']=os.pathsep.join(dll_dirs+[os.environ.get('PATH','')])
    from llama_cpp import Llama,llama_cpp
    path=Path(cfg['model_path'])
    with path.open('rb') as source:actual_hash=hashlib.file_digest(source,'sha256').hexdigest()
    if actual_hash!=cfg['model_sha256']:raise ValueError('GGUF hash mismatch')
    if not vocab_only and not llama_cpp.llama_supports_gpu_offload():raise RuntimeError('GPU offload unavailable; no automatic fallback')
    model=Llama(model_path=str(path),vocab_only=vocab_only,n_ctx=cfg['n_ctx'],n_gpu_layers=0 if vocab_only else cfg['n_gpu_layers'],
        n_threads=cfg['n_threads'],n_batch=cfg['n_batch'],n_ubatch=cfg['n_ubatch'],flash_attn=cfg['flash_attention'],seed=cfg['seed'],
        rope_scaling_type=cfg['rope_scaling_type'],rope_freq_scale=cfg['rope_freq_scale'],yarn_orig_ctx=cfg['yarn_orig_ctx'],verbose=True)
    template=model.metadata['tokenizer.chat_template']
    if hashlib.sha256(template.encode()).hexdigest()!=cfg['template_sha256']:model.close();raise ValueError('Native template hash mismatch')
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter,chat_formatter_to_chat_completion_handler
    formatter=Jinja2ChatFormatter(template=template,eos_token=model.detokenize([model.token_eos()],special=True).decode(),
        bos_token=model.detokenize([model.token_bos()],special=True).decode() if model.token_bos()>=0 else '',
        add_generation_prompt=True,stop_token_ids=[model.token_eos()])
    def fmt(**kw):kw['enable_thinking']=False;return formatter(**kw)
    model.chat_handler=chat_formatter_to_chat_completion_handler(fmt)
    return model,fmt

def messages(prep,assets):
    return [{'role':'system','content':(Path(assets)/'common_system_prompt.txt').read_text(encoding='utf-8')},
            {'role':'user','content':json.dumps(prep['logical_payload'],ensure_ascii=False,separators=(',',':'))}]

def preflight(prepared,assets,cfg,report):
    model,fmt=load_model(cfg,vocab_only=True);rows=[]
    try:
        manifest=read(Path(prepared)/'manifest.json')
        for item in manifest['items']:
            prep=read(Path(prepared)/item['file'])
            if digest(prep)!=item['prepared_hash']:raise ValueError('Prepared input changed')
            msg=messages(prep,assets);rendered=fmt(messages=msg).prompt
            n=len(model.tokenize(rendered.encode(),add_bos=False,special=True));required=n+cfg['max_output_tokens']+cfg['safety_tokens']
            rows.append({'paragraph_id':prep['paragraph_id'],'input_tokens':n,'output_reserved_tokens':cfg['max_output_tokens'],
                'safety_tokens':cfg['safety_tokens'],'required_tokens':required,'n_ctx':cfg['n_ctx'],'overage_tokens':max(0,required-cfg['n_ctx']),
                'fits':required<=cfg['n_ctx'],'messages_hash':digest(msg),'prepared_hash':digest(prep)})
        result={'passed':all(r['fits'] for r in rows),'mode':'native tokenizer capacity only; not a GPU loading or generation test',
                'configuration_hash':digest(cfg),'rows':rows}
        if Path(report).exists():raise FileExistsError('Capacity report exists')
        save(report,result);return result
    finally:model.close()

def run(prepared,output,assets,cfg,capacity_file):
    output=Path(output);manifest=read(Path(prepared)/'manifest.json');capacity=read(capacity_file)
    if not capacity['passed'] or capacity['configuration_hash']!=digest(cfg):raise ValueError('Missing/passed preflight for identical settings required')
    caps={r['paragraph_id']:r for r in capacity['rows']}
    asset_hash=digest({p.relative_to(assets).as_posix():p.read_bytes().hex() for p in sorted(Path(assets).rglob('*')) if p.is_file()})
    identity={'configuration_hash':digest(cfg),'manifest_hash':digest(manifest),'assets_hash':asset_hash,
              'code_hash':digest({p.relative_to(Path(__file__).parent).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.rglob('*.py'))})}
    if output.exists():
        if not (output/'identity.json').exists() or read(output/'identity.json')!=identity:raise ValueError('Output/configuration identity differs; choose a new run folder')
        if (output/'environment_error.json').exists():raise RuntimeError('Previous environment error requires explicit recovery; no automatic restart')
    else:output.mkdir(parents=True);save(output/'identity.json',identity)

    for item in manifest['items']:
        prep=read(Path(prepared)/item['file']);cap=caps.get(prep['paragraph_id'])
        if not cap or digest(prep)!=item['prepared_hash'] or cap['prepared_hash']!=digest(prep) or cap['messages_hash']!=digest(messages(prep,assets)):raise ValueError('Capacity/input/prompt identity differs')
    stop=threading.Event();samples=[]
    def monitor():
        while not stop.is_set():
            try:
                row={'elapsed_seconds':time.monotonic(),'nvidia_smi':subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,creationflags=0x08000000 if os.name=='nt' else 0).strip()}
                samples.append(row)
                with (output/'gpu_samples.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
            except Exception:pass
            stop.wait(1)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();model=None;done=[]
    criteria=read(Path(assets)/'criteria_v3_complete_reference.json')
    class Deadline(Exception):pass
    try:
        model,fmt=load_model(cfg)
        from llama_cpp import LlamaGrammar,llama_cpp
        save(output/'runtime.json',{'configuration':cfg,'actual_n_ctx':model.n_ctx(),'context_parameters':{
            k:getattr(model.context_params,k) for k in ('rope_scaling_type','rope_freq_scale','yarn_orig_ctx','yarn_ext_factor','yarn_attn_factor','yarn_beta_fast','yarn_beta_slow')},
            'thinking':False,'semantic_validity':'not_evaluated','note':'Actual offloaded layer count is recorded by llama.cpp in redirected native stderr; configuration alone is not proof of all layers loaded.'})
        for item in manifest['items']:
            prep=read(Path(prepared)/item['file']);case=output/'paragraphs'/Path(item['file']).stem;case.mkdir(parents=True,exist_ok=True)
            if (case/'result.json').exists():done.append(read(case/'result.json'));continue
            if (case/'attempt_started.json').exists():raise RuntimeError('Interrupted attempt found; preserve partial output and explicitly start a separately versioned run rather than auto-retry')
            msg=messages(prep,assets);save(case/'input.json',msg);save(case/'schema.json',prep['schema']);(case/'grammar.gbnf').write_text(prep['grammar'],encoding='utf-8')
            save(case/'attempt_started.json',{'paragraph_id':prep['paragraph_id'],'started_unix':time.time()})
            grammar=LlamaGrammar.from_string(prep['grammar'],verbose=False);model.reset();tokens=[];text='';end=None;finish=None;iterator=None;started=time.perf_counter();original=model.generate
            def observed(*args,**kw):
                nonlocal end
                for token in original(*args,**kw):
                    if not llama_cpp.llama_vocab_is_eog(model._model.vocab,token):tokens.append(token)
                    if time.perf_counter()-started>=cfg['per_case_wall_time_limit_seconds']:end='wall_time_limit_300s';raise Deadline()
                    yield token
            model.generate=observed
            try:
                iterator=model.create_chat_completion(messages=msg,grammar=grammar,temperature=cfg['temperature'],top_p=cfg['top_p'],top_k=cfg['top_k'],min_p=cfg['min_p'],
                    repeat_penalty=cfg['repeat_penalty'],presence_penalty=cfg['presence_penalty'],frequency_penalty=cfg['frequency_penalty'],max_tokens=cfg['max_output_tokens'],seed=cfg['seed'],stream=True)
                for chunk in iterator:
                    with (case/'chunks.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(chunk,ensure_ascii=False)+'\n')
                    choice=chunk['choices'][0];text+=choice.get('delta',{}).get('content','') or ''
                    (case/'partial.txt').write_text(text,encoding='utf-8');end=repeat_reason(text)
                    if end:break
                    if choice.get('finish_reason') is not None:finish=choice['finish_reason'];break
            except Deadline:pass
            finally:
                if iterator is not None:iterator.close()
                model.generate=original
            raw=None;parse_error=None
            try:raw=json.loads(text,object_pairs_hook=strict_object)
            except (ValueError,TypeError) as e:parse_error=str(e)
            complete=isinstance(raw,dict) and not end and not parse_error and raw.get('paragraph_id')==prep['paragraph_id']
            label=raw.get('proposed_label') if complete and raw.get('proposed_label') in ('C','S','O') else None
            val=validate(raw if isinstance(raw,dict) else {},prep['original_payload'],prep['document'],prep['registry'],prep['schema'],criteria)
            if not complete:val['validation_status']='not_completed';val['validation_errors'].append({'code':end or parse_error or 'incomplete_output'})
            review=label is not None and (val['validation_status']!='passed' or bool(raw.get('review_required')))
            result={'paragraph_id':prep['paragraph_id'],'doc_id':prep['doc_id'],'proposed_label':label,'display_status':'미분류' if label is None else '검토 필요' if review else label,
                'raw_output':raw,'raw_content':text,'validation':val,'input_tokens':caps[prep['paragraph_id']]['input_tokens'],
                'actual_output_tokens':len(tokens),'generation_seconds':time.perf_counter()-started,'finish_reason':finish,'effective_end_reason':end or finish,
                'used_case_ids':prep['retrieval']['used_case_ids'],'used_case_count':3,'truncated':False,'semantic_validity':'not_evaluated'}
            save(case/'result.json',result);done.append(result)
            save(output/'completed.json',{'paragraph_ids':[x['paragraph_id'] for x in done],'count':len(done),'total':len(manifest['items'])})
        bydoc={}
        for row in done:bydoc.setdefault(row['doc_id'],[]).append(row)
        save(output/'documents.json',{did:aggregate(rows) for did,rows in bydoc.items()})
        save(output/'status.json',{'status':'completed','responses':len(done),'expected':len(manifest['items']),'semantic_validity':'not_evaluated'})
    except BaseException as e:
        save(output/('user_interruption.json' if isinstance(e,KeyboardInterrupt) else 'environment_error.json'),{'error':repr(e),'completed_preserved':len(done),'automatic_retry':False})
        raise
    finally:
        stop.set();thread.join(timeout=3)
        if model is not None:model.close()

