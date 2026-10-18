
import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cso_rag.core import read,save,digest,prepare_one,documents,cases
from cso_rag.generation import run,messages

class Runner(unittest.TestCase):
    def test_save_and_resume_without_regenerating(self):
        cfg=read(ROOT/'config.example.json');doc=documents(read(ROOT/'examples/documents.json'))[0]
        prep=prepare_one(doc,doc['paragraphs'][0],cases(read(ROOT/'examples/retrieval_cases.json')),ROOT/'assets',1)
        prep['retrieval']={'used_case_ids':[x['case_id'] for x in prep['logical_payload']['cases']]}
        ids=sorted(e['evidence_id'] for e in prep['registry'] if e['namespace']=='E')
        raw={'paragraph_id':prep['paragraph_id'],'proposed_label':'S','applied_criteria':[{'criterion_id':'FOI-9-1-6','evidence_ids':ids,'satisfaction_reason':'연락처와 같은 대상의 이름이 연결된다.'}],
             'reason':'가상 실행 검사','review_required':False,'o_assessment':None}
        class Fake:
            calls=0
            context_params=types.SimpleNamespace(**{k:cfg.get(k,1) for k in ('rope_scaling_type','rope_freq_scale','yarn_orig_ctx','yarn_ext_factor','yarn_attn_factor','yarn_beta_fast','yarn_beta_slow')})
            def n_ctx(self):return cfg['n_ctx']
            def generate(self,*a,**kw):yield 1
            def reset(self):pass
            def close(self):pass
            def create_chat_completion(self,**kw):
                self.calls+=1
                yield {'choices':[{'delta':{'content':json.dumps(raw,ensure_ascii=False)},'finish_reason':'stop'}]}
        fake=Fake()
        module=types.SimpleNamespace(LlamaGrammar=types.SimpleNamespace(from_string=lambda *a,**kw:None),llama_cpp=types.SimpleNamespace(llama_vocab_is_eog=lambda *a:False))
        scratch=(ROOT/'runs').resolve();scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temp:
            assert Path(temp).resolve().is_relative_to(scratch)
            d=Path(temp);prepared=d/'prep';output=d/'run';capacity=d/'capacity.json'
            save(prepared/'items/000001.json',prep)
            save(prepared/'manifest.json',{'items':[{'file':'items/000001.json','paragraph_id':prep['paragraph_id'],'prepared_hash':digest(prep)}]})
            save(capacity,{'passed':True,'configuration_hash':digest(cfg),'rows':[{'paragraph_id':prep['paragraph_id'],'prepared_hash':digest(prep),'messages_hash':digest(messages(prep,ROOT/'assets')),'input_tokens':1}]})
            with patch.dict(sys.modules,{'llama_cpp':module}),patch('cso_rag.generation.load_model',return_value=(fake,None)),patch('cso_rag.generation.subprocess.check_output',side_effect=RuntimeError('monitor disabled for fixture')):
                run(prepared,output,ROOT/'assets',cfg,capacity)
                run(prepared,output,ROOT/'assets',cfg,capacity)
            self.assertEqual(fake.calls,1)
            result=read(output/'paragraphs/000001/result.json')
            self.assertEqual(result['proposed_label'],'S')
            self.assertEqual(result['validation']['validation_status'],'passed')
            self.assertEqual(read(output/'documents.json')[doc['doc_id']]['proposed_label'],'S')
            self.assertTrue((output/'paragraphs/000001/partial.txt').exists())

if __name__=='__main__':unittest.main()
