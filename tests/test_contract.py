import copy,json,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cso_rag.core import documents,cases,check_separation,prepare_one,read,aggregate
from cso_rag.validation import validate
from cso_rag.validation.validator import grammar_regex

class Contract(unittest.TestCase):
    def setUp(self):
        self.docs=documents(read(ROOT/'examples/documents.json'))
        self.cases=cases(read(ROOT/'examples/retrieval_cases.json'))
        self.prep=prepare_one(self.docs[0],self.docs[0]['paragraphs'][0],self.cases,ROOT/'assets',1)
    def test_allowlist_gold(self):
        bad=copy.deepcopy(self.docs);bad[0]['paragraphs'][0]['label']='S'
        with self.assertRaises(ValueError):documents(bad)
    def test_overlap(self):
        check_separation(self.docs,self.cases)
        bad=copy.deepcopy(self.cases);bad[0]['group_id']=self.docs[0]['group_id']
        with self.assertRaises(ValueError):check_separation(self.docs,bad)
    def test_offsets(self):
        bad=copy.deepcopy(self.docs);bad[0]['paragraphs'][0]['end']-=1
        with self.assertRaises(ValueError):documents(bad)
    def test_case_metadata_and_no_R_marks(self):
        p=self.prep
        self.assertEqual(len(p['logical_payload']['cases']),3)
        self.assertTrue(all('evidence_spans' in c and 'reason' in c for c in p['logical_payload']['cases']))
        self.assertTrue(all('[R:' not in c['annotated_text'] for c in p['logical_payload']['case_paragraphs']))
        self.assertFalse(p['criteria_selection']['gold_used'])
        self.assertFalse(p['criteria_selection']['retrieved_cases_used'])
    def raw(self):
        p=self.prep;e=[x['evidence_id'] for x in p['registry'] if x['namespace']=='E'];e.sort()
        return {'paragraph_id':p['paragraph_id'],'proposed_label':'S','applied_criteria':[{'criterion_id':'FOI-9-1-6','evidence_ids':e,'satisfaction_reason':'연락처와 같은 대상의 이름이 연결된다.'}],
                'reason':'가상 구조 검사. 의미상 정답 판정이 아님.','review_required':False,'o_assessment':None}
    def validation(self,raw):
        p=self.prep
        return validate(raw,p['original_payload'],p['document'],p['registry'],p['schema'],read(ROOT/'assets/criteria_v3_complete_reference.json'))
    def test_r1_id_restatement_diagnostic_only(self):
        val=self.validation(self.raw())
        self.assertEqual(val['validation_status'],'passed')
        self.assertTrue(val['validation_diagnostics'])
        self.assertFalse(val['semantic_validity_automatically_verified'])
    def test_source_rejection(self):
        raw=self.raw();raw['applied_criteria'][0]['evidence_ids']=['R:invented']
        self.assertEqual(self.validation(raw)['validation_status'],'failed')
    def test_grammar_branches_and_duplicates(self):
        grammar=grammar_regex(self.prep['grammar']);raw=self.raw()
        text=lambda x:json.dumps(x,ensure_ascii=False,separators=(',',':'))
        self.assertIsNotNone(grammar.fullmatch(text(raw)))
        raw['o_assessment']={'basis':'cs_non_applicable'}
        self.assertIsNone(grammar.fullmatch(text(raw)))
        raw=self.raw();raw['applied_criteria'][0]['evidence_ids']*=2
        self.assertIsNone(grammar.fullmatch(text(raw)))
    def test_O_empty_and_incomplete_document(self):
        p=self.prep;raw={'paragraph_id':p['paragraph_id'],'proposed_label':'O','applied_criteria':[],'reason':'구조 검사',
            'review_required':False,'o_assessment':{'basis':'cs_non_applicable','reviewed_paragraph_ids':[x['paragraph_id'] for x in [p['original_payload']['target'],*p['original_payload']['context']]],'cs_exclusion_summary':'구조 검사','uncertainty_note':''}}
        raw['o_assessment']['reviewed_paragraph_ids'].sort()
        self.assertEqual(self.validation(raw)['validation_status'],'passed')
        row={'proposed_label':None,'raw_output':None,'validation':{'validation_status':'not_completed'}}
        self.assertIsNone(aggregate([row])['proposed_label'])

if __name__=='__main__':unittest.main()
