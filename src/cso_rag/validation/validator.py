
import copy, json, re
from .validator_base import annotate, strict_object, validate as old_validate, repeat_reason as old_repeat

CRITERIA=['FOI-9-1-'+str(i) for i in range(1,9)]+['N2SF-S-LOG-BACKUP','O-PUBLIC','N2SF-O-RELEASED']
CIDS=CRITERIA[:4]; SIDS=CRITERIA[4:9]

def schema_for(base,pid,registry,grammar=False):
    s=copy.deepcopy(base);e=[r['evidence_id'] for r in registry if r['namespace']=='E'];t=[r['evidence_id'] for r in registry if r['namespace']=='E' and r['source_scope']=='target']
    sources=sorted({r['source_paragraph_id'] for r in registry if r['namespace']=='E'});target=next(r['source_paragraph_id'] for r in registry if r['source_scope']=='target')
    s['properties']['paragraph_id']={'type':'string','const':pid}
    item=s['properties']['applied_criteria']['items']
    item['properties']['evidence_ids'].update(items={'type':'string','enum':e},contains={'enum':t})
    for obj in [s['properties']['o_assessment']['anyOf'][0]]:
        obj['properties']['reviewed_paragraph_ids'].update(items={'type':'string','enum':sources},contains={'const':target})
    s['x-serialization-order']={'root_keys':s['required'],'criterion_ids':CRITERIA,'evidence_ids':sorted(e),'reviewed_paragraph_ids':sources,'no_automatic_sorting':True,'status':'additional executable contract, not standard JSON Schema keyword'}
    return s

def order_errors(raw):
    errors=[]
    claims=raw.get('applied_criteria',[]) if isinstance(raw,dict) else []
    if isinstance(claims,list) and all(isinstance(c,dict) for c in claims):
        ids=[c.get('criterion_id') for c in claims]
        if all(c in CRITERIA for c in ids) and ids!=sorted(set(ids),key=CRITERIA.index):errors.append({'code':'criterion_serialization_order_or_duplicate'})
        for i,c in enumerate(claims):
            e=c.get('evidence_ids')
            if isinstance(e,list) and all(isinstance(x,str) for x in e) and e!=sorted(set(e)):errors.append({'code':'evidence_serialization_order_or_duplicate','claim':i})
    oa=raw.get('o_assessment') if isinstance(raw,dict) else None
    if isinstance(oa,dict):
        p=oa.get('reviewed_paragraph_ids')
        if isinstance(p,list) and all(isinstance(x,str) for x in p) and p!=sorted(set(p)):errors.append({'code':'reviewed_source_order_or_duplicate'})
    return errors

def validate(raw,payload,doc,registry,schema,criteria):
    result=old_validate(raw,payload,doc,registry,schema,criteria)
    result['validation_errors'].extend(order_errors(raw))
    if result['validation_errors']:result['validation_status']='failed'
    return result

def repeat_reason(text):
    result=old_repeat(text)
    if result:return result
    ids=re.findall(r'"criterion_id"\s*:\s*"([^"]+)"',text)
    if len(ids)!=len(set(ids)):return 'duplicate_criterion_id_during_generation'
    return None

def make_grammar(pid,registry):





    rules={}
    def lit(s):return json.dumps(s,ensure_ascii=False)
    def value(x):return lit(json.dumps(x,ensure_ascii=False,separators=(',',':')))+' space'
    def key(k):return value(k)+' '+lit(':')+' space '
    def add(name,expr):rules[name]=expr;return name
    add('space',r'[ \t\n\r]*')
    add('char',r'[^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])')
    add('nonempty',lit('"')+' char+ '+lit('"')+' space')
    add('string',lit('"')+' char* '+lit('"')+' space')
    add('boolean','("true" | "false") space')
    def subsequence(name,choices,need=False):

        picks={};n=len(choices)
        for i in range(n-1,-1,-1):
            for required in [False,True]:
                expr,qualifies=choices[i];remaining=required and not qualifies;tail=[]
                if not remaining:tail.append(lit(''))
                nextpick=picks.get((i+1,remaining))
                if nextpick:tail.append(lit(',')+' space '+nextpick)
                alternatives=[]
                if tail:
                    tailname=add(f'{name}-tail-{i}-'+('need' if required else 'done'),' | '.join(tail))
                    alternatives.append(expr+' '+tailname)
                skip=picks.get((i+1,required))
                if skip:alternatives.append(skip)
                if alternatives:picks[i,required]=add(f'{name}-pick-{i}-'+('need' if required else 'done'),' | '.join(alternatives))
        assert (0,need) in picks
        return add(name,lit('[')+' space '+picks[0,need]+' '+lit(']')+' space')
    entries=sorted([r for r in registry if r['namespace']=='E'],key=lambda r:r['evidence_id'])
    evidence=subsequence('evidence',[(value(r['evidence_id']),r['source_scope']=='target') for r in entries],True)
    target=next(r['source_paragraph_id'] for r in entries if r['source_scope']=='target')
    reviewed=subsequence('reviewed',[(value(p),p==target) for p in sorted({r['source_paragraph_id'] for r in entries})],True)
    claims={}
    for i,cid in enumerate(CRITERIA):
        claims[cid]=add('claim-'+str(i),lit('{')+' space '+key('criterion_id')+value(cid)+' '+lit(',')+' space '+key('evidence_ids')+evidence+' '+lit(',')+' space '+key('satisfaction_reason')+'nonempty '+lit('}')+' space')
    cl=subsequence('criteria-c',[(claims[c],c in CIDS) for c in CIDS+SIDS],True)
    sl=subsequence('criteria-s',[(claims[c],True) for c in SIDS],True)
    empty=lit('[')+' space '+lit(']')+' space'
    ol=add('criteria-o-public',empty+' | '+lit('[')+' space '+claims['O-PUBLIC']+' '+lit(']')+' space')
    released=add('criteria-o-released',lit('[')+' space '+claims['N2SF-O-RELEASED']+' '+lit(']')+' space')
    def assessment(basis):
        return lit('{')+' space '+key('basis')+value(basis)+' '+lit(',')+' space '+key('reviewed_paragraph_ids')+reviewed+' '+lit(',')+' space '+key('cs_exclusion_summary')+'nonempty '+lit(',')+' space '+key('uncertainty_note')+'string '+lit('}')+' space'
    branches=[]
    for name,label,criteria,review,oa in [
        ('c','C',cl,'boolean',value(None)),('s','S',sl,'boolean',value(None)),
        ('o-public','O',ol,'boolean',assessment('cs_non_applicable')),
        ('o-released','O',released,'boolean',assessment('non_disclosure_expired')),
        ('unclassified',None,empty,value(True),value(None))]:
        expr=lit('{')+' space '+key('paragraph_id')+value(pid)+' '+lit(',')+' space '+key('proposed_label')+value(label)+' '+lit(',')+' space '+key('applied_criteria')+criteria+' '+lit(',')+' space '+key('reason')+'nonempty '+lit(',')+' space '+key('review_required')+review+' '+lit(',')+' space '+key('o_assessment')+oa+' '+lit('}')+' space'
        branches.append(add('root-'+name,expr))
    add('root',' | '.join(branches))
    return '\n'.join(name+' ::= '+expr for name,expr in rules.items())+'\n'

def grammar_regex(grammar):

    import regex
    definitions=[]
    for line in grammar.splitlines():
        name,expr=line.split(' ::= ',1);tokens=[];i=0
        while i<len(expr):
            c=expr[i]
            if c.isspace():i+=1;continue
            if c=='"':
                decoder=json.JSONDecoder();v,n=decoder.raw_decode(expr[i:]);tokens.append(regex.escape(v));i+=n
            elif c=='[':
                j=i+1
                while j<len(expr):
                    if expr[j]=='\\':j+=2;continue
                    if expr[j]==']':break
                    j+=1
                tokens.append(expr[i:j+1]);i=j+1
            elif c.isalpha():
                m=re.match(r'[a-zA-Z][a-zA-Z0-9-]*',expr[i:]);v=m.group();tokens.append('(?&'+v.replace('-','_')+')');i+=len(v)
            else:tokens.append(c);i+=1
        definitions.append('(?P<'+name.replace('-','_')+'>'+''.join(tokens)+')')
    return regex.compile('(?(DEFINE)'+''.join(definitions)+')(?&root)')
