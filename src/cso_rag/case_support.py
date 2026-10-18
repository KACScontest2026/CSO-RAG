
import copy

def deduplicate_payload(payload):
    out=copy.deepcopy(payload); nodes={};removed=[];case_seen=set();newcases=[]
    def add(pid,text,doc_id):
        if pid in nodes:
            if nodes[pid]['text']!=text or nodes[pid]['doc_id']!=doc_id:raise ValueError('Conflicting paragraph ID')
            removed.append({'paragraph_id':pid,'reason':'same_id_same_text_already_stored'})
        else:nodes[pid]={'paragraph_id':pid,'doc_id':doc_id,'text':text}
    for c in out['cases']:
        if c['case_id'] in case_seen:removed.append({'case_id':c['case_id'],'reason':'duplicate_case_id'});continue
        case_seen.add(c['case_id']);add(c['paragraph_id'],c['text'],c['doc_id'])
        ids=[]
        for ctx in c.pop('case_context',[]):
            add(ctx['paragraph_id'],ctx['text'],c['doc_id'])
            if ctx['paragraph_id']!=c['paragraph_id'] and ctx['paragraph_id'] not in ids:ids.append(ctx['paragraph_id'])
        c.pop('text');c['body_paragraph_id']=c['paragraph_id'];c['context_paragraph_ids']=ids;newcases.append(c)
    out['cases']=newcases;out['case_paragraphs']=list(nodes.values())
    contexts={};target=out['target']
    for p in out['context']:
        if p['paragraph_id']==target['paragraph_id']:
            if p['text']!=target['text']:raise ValueError('Conflicting target ID')
            removed.append({'paragraph_id':p['paragraph_id'],'reason':'target_already_provided'});continue
        if p['paragraph_id'] in contexts:
            if contexts[p['paragraph_id']]!=p:raise ValueError('Conflicting context ID')
            removed.append({'paragraph_id':p['paragraph_id'],'reason':'duplicate_target_context'})
        else:contexts[p['paragraph_id']]=p
    out['context']=list(contexts.values())
    assert out['target']==payload['target'] and out['criteria']==payload['criteria'] and out['workflow_context']==payload['workflow_context']
    for c in payload['cases']:
        assert nodes[c['paragraph_id']]['text']==c['text']
        for ctx in c.get('case_context',[]):assert nodes[ctx['paragraph_id']]['text']==ctx['text']
    return out,removed
