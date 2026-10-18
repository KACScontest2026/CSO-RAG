
import json, time
from pathlib import Path
from .core import read,save,cases,documents,check_separation,prepare_one,digest

def pooling_modes(model):

    modes=[]
    legacy=('cls_token','max_tokens','mean_tokens','mean_sqrt_len_tokens','weightedmean_tokens','lasttoken')
    names={'cls_token':'cls','max_tokens':'max','mean_tokens':'mean','weightedmean_tokens':'weightedmean'}
    for module in model:
        mode=getattr(module,'pooling_mode',None)
        if mode is not None:
            modes.extend([mode] if isinstance(mode,str) else list(mode))
        elif hasattr(module,'pooling_mode_lasttoken'):
            modes.extend(names.get(name,name) for name in legacy if getattr(module,'pooling_mode_'+name,False))
    return modes

def encoder(cfg):
    import torch
    from sentence_transformers import SentenceTransformer
    if cfg['embedding_device']=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=SentenceTransformer(cfg['embedding_model'],revision=cfg['embedding_revision'],device=cfg['embedding_device'],
        model_kwargs={'torch_dtype':torch.float32,'attn_implementation':'sdpa'},trust_remote_code=False)
    model.max_seq_length=cfg['embedding_max_length'];model.tokenizer.padding_side='left'
    if pooling_modes(model)!=['lasttoken']:raise RuntimeError('Expected Qwen last-token pooling')
    if model.get_sentence_embedding_dimension()!=cfg['embedding_dimension']:raise RuntimeError('Embedding dimension mismatch')
    return model

def vectorize(model,texts,cfg,query=False):
    import numpy as np
    prefix=cfg['query_prefix'] if query else ''
    rendered=[prefix+t for t in texts]
    lengths=[len(model.tokenizer(t,add_special_tokens=True,truncation=False)['input_ids']) for t in rendered]
    if max(lengths)>cfg['embedding_max_length']:raise ValueError('Embedding input exceeds limit; no truncation allowed')
    start=time.perf_counter()
    v=model.encode(rendered,batch_size=1,normalize_embeddings=True,convert_to_numpy=True,prompt='',show_progress_bar=False).astype(np.float32)
    if v.shape!=(len(texts),cfg['embedding_dimension']) or not np.isfinite(v).all() or not np.allclose(np.linalg.norm(v,axis=1),1,atol=1e-4):raise ValueError('Invalid embedding vectors')
    return v,{'seconds':time.perf_counter()-start,'input_tokens':lengths,'truncated':False,'device':str(model.device),'dimension':v.shape[1],
              'pooling_modes':pooling_modes(model),'finite_values':True,'l2_norms':np.linalg.norm(v,axis=1).tolist(),'dtype':str(v.dtype)}

def client(path):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(Path(path).resolve()),settings=Settings(anonymized_telemetry=False))

def build_index(data,index,cfg):
    index=Path(index)
    if index.exists():raise FileExistsError('Use a new index directory; existing indices are never overwritten')
    records=cases(read(data));model=encoder(cfg)
    start=time.perf_counter();v,audit=vectorize(model,[c['text'] for c in records],cfg)
    index.mkdir(parents=True)
    col=client(index).create_collection(cfg['collection'],embedding_function=None,metadata={'hnsw:space':'cosine'})
    for offset in range(0,len(records),256):
        batch=records[offset:offset+256]
        col.add(ids=[c['case_id'] for c in batch],embeddings=v[offset:offset+256].tolist(),
                metadatas=[{'record':json.dumps(c,ensure_ascii=False)} for c in batch])
    save(index/'manifest.json',{'configuration':cfg,'cases_hash':digest(records),'records':len(records),
         'documents':len({c['doc_id'] for c in records}),'paragraphs':len({c['paragraph_id'] for c in records}),
         'body_hashes':{c['case_id']:digest(c['text']) for c in records},'embedding_audit':audit,'index_build_seconds':time.perf_counter()-start})

def prepare(input_file,index,output,assets,cfg):
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new preparation directory')
    docs=documents(read(input_file));manifest=read(Path(index)/'manifest.json')
    keys=('embedding_model','embedding_revision','embedding_dimension','embedding_max_length','query_prefix')
    if any(cfg[k]!=manifest['configuration'][k] for k in keys):raise ValueError('Embedding/index configuration mismatch')
    col=client(index).get_collection(cfg['collection'],embedding_function=None)
    records=[json.loads(x['record']) for x in col.get(include=['metadatas'])['metadatas']]
    cases(records);check_separation(docs,records)
    model=encoder(cfg);rows=[];ordinal=0
    for doc in docs:
        for p in doc['paragraphs']:
            ordinal+=1;v,embedding_audit=vectorize(model,[p['text']],cfg,query=True)
            start=time.perf_counter();hit=col.query(query_embeddings=v.tolist(),n_results=3,include=['metadatas','distances'])
            elapsed=time.perf_counter()-start;selected=[json.loads(x['record']) for x in hit['metadatas'][0]]
            if len(selected)!=3 or len(set(hit['ids'][0]))!=3:raise ValueError('Three distinct cases required')
            prep=prepare_one(doc,p,selected,assets,ordinal)
            prep['retrieval']={'retrieved_ids':hit['ids'][0],'used_case_ids':hit['ids'][0],'used_case_count':3,
                'ranked_cases':[{'rank':i+1,'case_id':cid,'cosine_distance':dist,'cosine_similarity':1-dist} for i,(cid,dist) in enumerate(zip(hit['ids'][0],hit['distances'][0]))],
                'query_embedding':embedding_audit,'search_seconds':elapsed,'index_cases_hash':manifest['cases_hash']}
            filename=f'{ordinal:06d}.json';save(output/'items'/filename,prep)
            rows.append({'file':'items/'+filename,'paragraph_id':p['paragraph_id'],'doc_id':doc['doc_id'],'prepared_hash':digest(prep)})
    save(output/'manifest.json',{'condition':'evidence_rag','documents':len(docs),'paragraphs':len(rows),'items':rows,
         'configuration':cfg,'gold_used':False,'index_records':manifest['records'],'index_documents':manifest['documents'],
         'criteria_assets_hash':digest({p.relative_to(assets).as_posix():p.read_bytes().hex() for p in sorted(Path(assets).rglob('*')) if p.is_file()})})
