import argparse,json
from pathlib import Path
from .core import read

def main():
    p=argparse.ArgumentParser(description='Evidence-enriched RAG only; no A/B comparison modes')
    p.add_argument('--config',default='config.example.json');p.add_argument('--assets',default='assets')
    commands=p.add_subparsers(dest='command',required=True)
    q=commands.add_parser('check-input');q.add_argument('--input',required=True);q.add_argument('--cases',required=True)
    q=commands.add_parser('build-index');q.add_argument('--cases',required=True);q.add_argument('--index',required=True)
    q=commands.add_parser('prepare');q.add_argument('--input',required=True);q.add_argument('--index',required=True);q.add_argument('--output',required=True)
    q=commands.add_parser('capacity');q.add_argument('--prepared',required=True);q.add_argument('--report',required=True)
    q=commands.add_parser('run');q.add_argument('--prepared',required=True);q.add_argument('--output',required=True);q.add_argument('--capacity',required=True)
    args=p.parse_args();cfg=read(args.config);assets=Path(args.assets)
    if cfg.get('condition')!='evidence_rag' or cfg['retrieved_case_count']!=3:raise ValueError('This package supports evidence RAG with exactly three cases only')
    if args.command=='check-input':
        from .core import documents,cases,check_separation
        docs=documents(read(args.input));records=cases(read(args.cases));check_separation(docs,records)
        print(json.dumps({'passed':True,'evaluation_documents':len(docs),'evaluation_paragraphs':sum(len(d['paragraphs']) for d in docs),'retrieval_cases':len(records),'gold_fields_allowed':False,'source_relationship_scope':'provided group IDs and exact paragraph equality only; translations/derivatives require prior source audit'}))
    elif args.command=='build-index':
        from .retrieval import build_index
        build_index(args.cases,args.index,cfg)
    elif args.command=='prepare':
        from .retrieval import prepare
        prepare(args.input,args.index,args.output,assets,cfg)
    elif args.command=='capacity':
        from .generation import preflight
        report=preflight(args.prepared,assets,cfg,args.report);print(json.dumps({'passed':report['passed'],'paragraphs':len(report['rows'])}))
    else:
        from .generation import run
        run(args.prepared,args.output,assets,cfg,args.capacity)

if __name__=='__main__':main()
