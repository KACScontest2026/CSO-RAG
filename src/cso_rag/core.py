
import copy
import hashlib
import json
from pathlib import Path
from .case_support import deduplicate_payload
from .criteria import criteria_engine
from .validation.validator_base import annotate
from .validation import schema_for, make_grammar

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(path)

def documents(raw):

    if not isinstance(raw,list) or not raw:raise ValueError('Expected a nonempty document list')
    result=[];dids=set();pids=set()
    for d in raw:
        if set(d)!={'doc_id','group_id','text','paragraphs'}:raise ValueError('Document fields must be doc_id, group_id, text, paragraphs; no gold fields')
        if not all(isinstance(d[k],str) and d[k] for k in ('doc_id','group_id','text')):raise ValueError('Document IDs, group and text must be nonempty strings')
        if d['doc_id'] in dids:raise ValueError('Duplicate document ID')
        dids.add(d['doc_id']);previous=-1
        if not d['paragraphs']:raise ValueError('No paragraphs')
        for p in d['paragraphs']:
            if set(p)!={'paragraph_id','text','start','end'}:raise ValueError('Paragraph allowlist violated; no gold fields')
            if not isinstance(p['paragraph_id'],str) or not p['paragraph_id'] or p['paragraph_id'] in pids:raise ValueError('Missing/duplicate paragraph ID')
            if not isinstance(p['text'],str) or not p['text']:raise ValueError('Empty paragraph')
            if type(p['start']) is not int or type(p['end']) is not int or not (previous<p['start']<=p['end']<len(d['text'])):raise ValueError('Invalid/overlapping positions')
            if d['text'][p['start']:p['end']+1]!=p['text']:raise ValueError('Paragraph/source mismatch; Python Unicode inclusive offsets required')
            previous=p['end'];pids.add(p['paragraph_id'])
        result.append(copy.deepcopy(d))
    return result

def cases(raw):
    required={'case_id','paragraph_id','doc_id','group_id','text','label','evidence_spans','criterion_ids','reason','case_context'}
    out=[];ids=set();bypara={}
    def source(pid,text,did):
        key=(did,text)
        if pid in bypara and bypara[pid]!=key:raise ValueError('Conflicting retrieval paragraph ID')
        bypara[pid]=key
    for c in raw:
        if set(c)!=required:raise ValueError('Retrieval case fields differ from documented contract')
        for k in ('case_id','paragraph_id','doc_id','group_id','text','reason'):
            if not isinstance(c[k],str) or not c[k]:raise ValueError('Empty case field: '+k)
        if c['case_id'] in ids:raise ValueError('Duplicate case ID')
        ids.add(c['case_id']);source(c['paragraph_id'],c['text'],c['doc_id'])
        if c['label'] not in ('C','S','O'):raise ValueError('Invalid case label')
        from .validation.validator import CRITERIA
        if not isinstance(c['criterion_ids'],list) or any(x not in CRITERIA for x in c['criterion_ids']):raise ValueError('Unconfirmed criterion alias; supply reviewed canonical IDs, preserving original IDs in separate history')
        if not isinstance(c['case_context'],list) or not isinstance(c['evidence_spans'],list):raise ValueError('Case evidence/context must be lists')
        nodes={c['paragraph_id']:c['text']}
        for p in c['case_context']:
            if set(p)!={'paragraph_id','text'}:raise ValueError('Linked context allowlist violated')
            source(p['paragraph_id'],p['text'],c['doc_id']);nodes[p['paragraph_id']]=p['text']
        for e in c['evidence_spans']:
            if set(e)!={'source_paragraph_id','source_scope','start','end','text'}:raise ValueError('Evidence fields differ from contract')
            s=nodes.get(e['source_paragraph_id']);a=e['start'];b=e['end']
            if s is None or type(a) is not int or type(b) is not int or not (0<=a<=b<len(s)) or s[a:b+1]!=e['text']:raise ValueError('Case evidence/source/position mismatch')
        if c['label'] in ('C','S') and not c['criterion_ids']:raise ValueError('C/S case needs reviewed criteria')
        from .validation.validator import CIDS,SIDS
        mapped=['C' if cid in CIDS else 'S' if cid in SIDS else 'O' for cid in c['criterion_ids']]
        if mapped and max(mapped,key={'O':0,'S':1,'C':2}.get)!=c['label']:raise ValueError('Retrieval case grade/criterion mismatch; do not relabel automatically')
        out.append(copy.deepcopy(c))
    if len(out)<3:raise ValueError('At least three cases required')
    return out

def check_separation(docs, records):
    dids={d['doc_id'] for d in docs};groups={d['group_id'] for d in docs}
    texts={p['text'] for d in docs for p in d['paragraphs']}
    for c in records:
        if c['doc_id'] in dids or c['group_id'] in groups:raise ValueError('Retrieval/evaluation document or source-group overlap')
        if c['text'] in texts or any(p['text'] in texts for p in c['case_context']):raise ValueError('Exact paragraph duplicate across retrieval/evaluation')

def prepare_one(doc,target,hits,assets,ordinal):
    criteria,audit=criteria_engine(assets)([target,*[p for p in doc['paragraphs'] if p['paragraph_id']!=target['paragraph_id']]])
    payload={'target':target,'context':[p for p in doc['paragraphs'] if p['paragraph_id']!=target['paragraph_id']],
             'criteria':criteria,'workflow_context':read(Path(assets)/'workflow_context.json'),'cases':copy.deepcopy(hits)}
    payload,removed=deduplicate_payload(payload)

    fields=('case_id','paragraph_id','doc_id','label','body_paragraph_id','context_paragraph_ids','evidence_spans','criterion_ids','reason')
    payload['cases']=[{k:c[k] for k in fields if k in c} for c in payload['cases']]
    marked,registry=annotate(payload,doc,ordinal)

    for p in marked['case_paragraphs']:
        p['annotated_text']=next(x['text'] for x in payload['case_paragraphs'] if x['paragraph_id']==p['paragraph_id'])
    schema=schema_for(read(Path(assets)/'llm_output_schema.json'),target['paragraph_id'],registry)
    return {'doc_id':doc['doc_id'],'paragraph_id':target['paragraph_id'],'document':doc,
            'original_payload':payload,'logical_payload':marked,'registry':registry,
            'schema':schema,'grammar':make_grammar(target['paragraph_id'],registry),
            'criteria_selection':audit,'deduplication_log':removed,'truncated':False,'case_exclusions':[]}

def aggregate(rows):
    labels=[r['proposed_label'] for r in rows]
    valid=[x for x in labels if x in ('C','S','O')]
    partial=max(valid,key={'O':0,'S':1,'C':2}.get) if valid else None
    complete=len(valid)==len(rows) and bool(rows)
    return {'complete':complete,'proposed_label':partial if complete else None,'partial_proposed_label':partial,
            'mandatory_validation_passed':complete and all(r['validation']['validation_status']=='passed' for r in rows),
            'model_review_requested':any(r['raw_output'] and r['raw_output'].get('review_required') for r in rows),
            'semantic_validity':'not_evaluated'}
