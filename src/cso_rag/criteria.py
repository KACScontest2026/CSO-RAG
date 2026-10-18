
import json, hashlib, re
from pathlib import Path
def read(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def h(s):return hashlib.sha256(s.encode()).hexdigest()
def compact(t):return re.sub(r'\s+','',t)

def criteria_engine(assets):
 F=Path(assets)
 full=read(F/'criteria_reference_v3/integrated_criteria_v3.json');rules=read(F/'criteria_input/selection_rules.json');core=read(F/'criteria_input/mandatory_core.json');catalog=read(F/'criteria_input/source_catalog.json');coretext=(F/'criteria_input/mandatory_core_model_input.txt').read_text(encoding='utf-8')
 institutions=full['domains']['기관별 공식 세부기준'];laws=[x for x in full['domains']['법령 원문'] if x['record_id'].startswith('LAW-IND-')];sourceby={x['record_id']:x for xs in full['domains'].values() for x in xs};bindings=catalog['record_source_bindings'];sources=catalog['source_versions'];lexicon=rules['institution_lexicon'];df=rules['source_document_frequency'];triggers=rules['individual_law_triggers'];corekeys={bindings[x['record_id']] for x in core['legal_conditions_and_exceptions']+core['n2sf_mapping']}
 def select(allowed):
  text='\n'.join(x['text'] for x in allowed);ct=compact(text);selected={};decisions=[]
  def hitloc(term):return [{'paragraph_id':x['paragraph_id'],'raw_match':term} for x in allowed if compact(term) in compact(x['text'])]
  for x in institutions:
   rid=x['record_id'];task=x.get('task') or '';department=x.get('department') or '';task_hits=[p for p in [task,department] if len(compact(p))>=3 and compact(p) in ct];terms=[w for w in lexicon[rid] if w in ct];rare=[w for w in terms if df[w]<=20];law_hits=[law['source'] for law in laws if compact(law['source']) in compact(x['official_article_cell']) and compact(law['source']) in ct]
   chosen=bool(task_hits or (len(terms)>=2 and rare) or law_hits);reason={'task_or_department_matches':task_hits,'content_terms':terms,'rare_terms':rare,'explicit_cited_law_matches':law_hits}
   if chosen:selected[rid]=(x,reason)
   decisions.append({'record_id':rid,'selected':chosen,'selection_reason':reason,'exclusion_reason':None if chosen else 'frozen lexical task/content/law relevance condition not met; not a finding of legal non-applicability','uncertain':True,'institution_application_scope':'행정안전부 해당 업무/부서에 한정; 대상 기관·정보 성격의 적합성은 별도 판정','MOIS_name_present_in_provided_text':'행정안전부' in text,'match_locations':[loc for term in task_hits+terms+law_hits for loc in hitloc(term)]})
  for x in laws:
   rid=x['record_id'];hits=[w for w in triggers[rid]+[x['source']] if compact(w) in ct];linked=[r for r,(node,_) in selected.items() if node.get('domain')=='기관별 공식 세부기준' and compact(x['source']) in compact(node['official_article_cell'])];chosen=bool(hits or linked);reason={'raw_text_triggers':hits,'selected_institution_citation_ids':linked}
   if chosen:selected[rid]=(x,reason)
   decisions.append({'record_id':rid,'selected':chosen,'selection_reason':reason,'exclusion_reason':None if chosen else 'no predefined raw-text/law-name trigger or selected institution citation; not a finding of non-applicability','uncertain':True,'match_locations':[loc for term in hits for loc in hitloc(term)]})
  lines=[coretext,'\n[원문·실제 제공 문맥으로 선택한 추가 참조: 적용·선택 불확실]']
  for rid,(x,_) in selected.items():
   lines+=['\n'+rid+' | '+bindings[rid]+' | '+x['original_location']]
   if x['domain']=='기관별 공식 세부기준':lines+=['업무: '+str(x.get('task') or '')+' / 부서: '+str(x.get('department') or '')+' / '+x['section'],'공식 세부기준 원문: '+x['official_detail_text'],'공식 관련근거 셀: '+x['official_article_cell'],'상위 기준 ID: '+', '.join(x['parent_criterion_ids']),'확인 상태: '+x['verification_status']]
   else:lines+=['역할: '+str(x.get('role') or ''),'법령 원문(조건·예외 포함): '+x['original_text'],'프로젝트 적용: '+str(x.get('project_application') or ''),'확인 필요: '+str(x.get('remaining_check') or ''),'확인 상태: '+x['verification_status']]
  extras={bindings[rid] for rid in selected}-corekeys;lines+=['출처·버전 참조(선택된 항목):']+[k+' | '+str(v[0])+' | '+str(v[1]) for k,v in sources.items() if k in extras];rendered='\n'.join(lines)
  for x in core['legal_conditions_and_exceptions']:assert x['original_text'] in rendered
  for x in core['project_instructions']:assert x['instruction'] in rendered
  for rid,(x,_) in selected.items():assert (x['official_detail_text'] if x['domain']=='기관별 공식 세부기준' else x['original_text']) in rendered
  return rendered,{'selected_ids':list(selected),'decisions':decisions,'selection_uncertain':True,'gold_used':False,'retrieved_cases_used':False,'mandatory_core_sha256':h(coretext),'criteria_payload_sha256':h(rendered),'raw_source_sha256':h(json.dumps(allowed,ensure_ascii=False,sort_keys=True))}
 return select
