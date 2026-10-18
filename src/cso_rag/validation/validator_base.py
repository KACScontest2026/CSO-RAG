
import copy,json,re
from jsonschema import Draft202012Validator
RANK={'O':0,'S':1,'C':2}
def segments(text):
    bounds={0,len(text)}
    for m in re.finditer(r'\n+|[.!?。？！](?=\s|$)',text):

        if m.group() == '.' and re.fullmatch(r'\s*(?:[-*]\s*)?(?:\d+|[a-zA-Z])',text[text.rfind('\n',0,m.start())+1:m.start()]):continue
        bounds.add(m.end())
    intervals=[];pending=0
    for a,b in zip(sorted(bounds),sorted(bounds)[1:]):
        s=text[pending:b];clean=s.strip()
        heading=(len(clean)<=80 and not re.search(r'[.!?。？！]$',clean) and not '|' in clean)
        if not clean or heading:continue
        intervals.append([pending,b]);pending=b
    if pending<len(text):
        if intervals:intervals[-1][1]=len(text)
        else:intervals.append([0,len(text)])
    assert ''.join(text[a:b] for a,b in intervals)==text
    return [{'start':a,'end':b-1,'text':text[a:b]} for a,b in intervals]
def annotate(payload,doc,ordinal):
    marked=copy.deepcopy(payload);registry=[];source={p['paragraph_id']:p for p in doc['paragraphs']};order={p['paragraph_id']:i+1 for i,p in enumerate(doc['paragraphs'])}
    def mark(p,ns,num,scope):
        entries=[]
        for j,span in enumerate(segments(p['text']),1):
            eid=f'{ns}:d{ordinal:02d}:p{num:02d}:u{j:03d}'
            rec={**span,'evidence_id':eid,'namespace':ns,'source_paragraph_id':p['paragraph_id'],'source_scope':scope}
            if ns=='E':
                assert p==source[p['paragraph_id']]
                rec.update(document_start=p['start']+span['start'],document_end=p['start']+span['end'])
                assert doc['text'][rec['document_start']:rec['document_end']+1]==rec['text']
            registry.append(rec);entries.append('['+eid+']'+span['text'])
        return {k:v for k,v in p.items() if k!='text'}|{'annotated_text':''.join(entries)}
    marked['target']=mark(payload['target'],'E',order[payload['target']['paragraph_id']],'target')
    marked['context']=[mark(p,'E',order[p['paragraph_id']],'provided_context') for p in payload['context']]
    marked['case_paragraphs']=[mark(p,'R',i+1,'retrieval_reference') for i,p in enumerate(payload['case_paragraphs'])]
    assert len({x['evidence_id'] for x in registry})==len(registry)
    return marked,registry
def schema_for(base,pid,registry,grammar=False):
    s=copy.deepcopy(base);s['properties']['paragraph_id']={'const':pid,'type':'string'}
    s['properties']['applied_criteria']['items']['properties']['evidence_ids']['items']={'type':'string','enum':[r['evidence_id'] for r in registry if r['namespace']=='E']}
    return s
def strict_object(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate_json_key:'+k)
        out[k]=v
    return out
def validate(raw,payload,doc,registry,schema,criteria):
    diagnostics=[]
    errors=[{'code':'json_schema_validation','detail':e.message,'path':list(e.path)} for e in Draft202012Validator(schema).iter_errors(raw)]
    byid={r['evidence_id']:r for r in registry};known={c['criterion_id']:c['label'] for c in criteria};sources={p['paragraph_id']:p for p in [payload['target'],*payload['context']]};target=payload['target']['paragraph_id'];restored=[];seen=set();claims=[]
    if not isinstance(raw,dict):return {'validation_status':'failed','validation_errors':errors+[{'code':'not_object'}],'evidence_spans':[],'criterion_groundings':[]}
    for i,claim in enumerate(raw.get('applied_criteria',[]) if isinstance(raw.get('applied_criteria'),list) else []):
        if not isinstance(claim,dict):continue
        cid=claim.get('criterion_id');eids=claim.get('evidence_ids',[]);eids=eids if isinstance(eids,list) else [];spans=[]
        if cid in seen:errors.append({'code':'duplicate_criterion_id','claim':i})
        seen.add(cid)
        if len(set(str(x) for x in eids))!=len(eids):errors.append({'code':'duplicate_evidence_id','claim':i})
        for eid in eids:
            entry=byid.get(eid) if isinstance(eid,str) else None
            if not entry or entry['namespace']!='E':errors.append({'code':'unknown_or_non_evaluation_evidence_id','claim':i});continue
            p=sources.get(entry['source_paragraph_id'])
            if not p:errors.append({'code':'source_not_provided','claim':i});continue
            a,b=entry['start'],entry['end'];ga,gb=entry['document_start'],entry['document_end']
            if p['text'][a:b+1]!=entry['text'] or doc['text'][ga:gb+1]!=entry['text'] or ga!=p['start']+a or gb!=p['start']+b:raise RuntimeError('Evidence registry corrupted')
            spans.append(copy.deepcopy(entry))
        if not any(s['source_paragraph_id']==target for s in spans):errors.append({'code':'missing_target_anchor','claim':i})
        context_ids=[s['evidence_id'] for s in spans if s['source_paragraph_id']!=target]
        reason=claim.get('satisfaction_reason','')
        if context_ids and (not isinstance(reason,str) or not any(e in reason for e in context_ids) or not any(s['evidence_id'] in reason for s in spans if s['source_paragraph_id']==target)):
            diagnostics.append({'code':'context_relation_IDs_not_explicit','claim':i,'mandatory_failure':False,'causes_review_required':False,'semantic_validity':'not_evaluated'})
        claims.append({**claim,'evidence_spans':spans});restored.extend(spans)
    labels=[known[c] for c in seen if c in known]
    if labels and raw.get('proposed_label')!=max(labels,key=RANK.get):errors.append({'code':'criterion_label_mismatch'})
    if raw.get('proposed_label')=='O':
        oa=raw.get('o_assessment')
        if isinstance(oa,dict) and any(pid not in sources for pid in oa.get('reviewed_paragraph_ids',[])):errors.append({'code':'O_reviewed_source_not_provided'})

    union={s['evidence_id']:s for s in restored}
    return {'validation_status':'failed' if errors else 'passed','validation_errors':errors,'evidence_spans':list(union.values()),'criterion_groundings':claims,'semantic_validity_automatically_verified':False,'validation_diagnostics':diagnostics}
def repeat_reason(text):
    for m in re.finditer(r'"evidence_ids"\s*:\s*\[([^\]]*)',text,re.S):
        eids=re.findall(r'"(E:[^"\\]+)"',m.group(1))
        if len(eids)!=len(set(eids)):return 'duplicate_evidence_id_within_criterion'
    if len(text)>=512 and not text[-512:].strip():return 'whitespace_no_progress'
    for n in [64,96,128]:
        if len(text)>=3*n and text[-n:]==text[-2*n:-n]==text[-3*n:-2*n]:return 'consecutive_text_loop'
    return None
