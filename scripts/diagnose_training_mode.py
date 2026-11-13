"""Paired exact-trainer toy audit: archived release versus isolated fix.

Synthetic inputs and tiny random models are diagnostics, never project scores.
"""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import torch
from transformers import GPT2Config,GPT2LMHeadModel

from toksw.data import encode_corpus,iter_jsonl
from toksw.metrics import bits_per_byte
from toksw.tok import load_tokenizer,train_bpe


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--old-root',type=Path,required=True)
    ap.add_argument('--fixed-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--scratch',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mode-audit-toy-',dir=args.scratch) as d:
        root=Path(d)
        rng=random.Random(2026)
        words=['def','return','self','data','model','token','中文','函数','语言']
        texts=[' '.join(rng.choices(words,k=16))+'\n' for _ in range(100)]
        raw=root/'raw/code';raw.mkdir(parents=True)
        train=raw/'train.jsonl'
        train.write_text(''.join(json.dumps(dict(text=t),ensure_ascii=False)+'\n' for t in texts))
        for split,text in [('validation','validation only 中文'),('test','held-out toy 中文 test')]:
            (raw/f'{split}.jsonl').write_text(json.dumps(dict(text=text))+'\n')
        old,new=train_bpe(texts,300,'gpt2',min_frequency=1),train_bpe(texts,400,'gpt4',min_frequency=1)
        for name,tok in [('old',old),('new',new)]:
            tok.save(str(root/f'{name}.json'))
            encode_corpus(tok,train,root/'tokens/code'/name)
        torch.manual_seed(2026)
        base=root/'base'
        GPT2LMHeadModel(GPT2Config(vocab_size=old.get_vocab_size(),n_positions=32,n_embd=16,
                                  n_layer=1,n_head=2,attn_pdrop=.1,embd_pdrop=.1,resid_pdrop=.1)).save_pretrained(base)
        traces,comparisons={},[]
        for condition in ['A_keep','B_swap_fvt','E_progressive']:
            weights={}
            for label,source in [('release',args.old_root),('fixed',args.fixed_root)]:
                trace=root/f'{condition}_{label}.json'
                out=root/'runs'/label
                cmd=[sys.executable,str(args.fixed_root/'scripts/trace_training_mode.py'),
                     '--script',str(source/'scripts/train.py'),'--out',str(trace),'--',
                     f'configs/train/{condition}.yaml',f'base_model={base}',f'old_tokenizer={root/"old.json"}',
                     f'new_tokenizer={root/"new.json"}','old_name=old','new_name=new',f'raw_dir={raw.parent}',
                     f'tokens_dir={root/"tokens"}',f'out={out}',f'run_name={condition}','token_budget=128',
                     'block_size=32','micro_batch=1','grad_accum=1','eval_every=2','eval_docs=1',
                     'eval_batch=1','eval_split=validation','log_every=1']
                env=os.environ|dict(PYTHONPATH=str(source/'src'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                                     OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
                subprocess.run(cmd,cwd=source,env=env,check=True,capture_output=True,text=True,timeout=180)
                t=json.loads(trace.read_text())
                train_events=t['training_forwards'];eval_events=t['evaluation_forwards']
                assert train_events and eval_events and all(not e['model_training'] for e in eval_events)
                assert all(e['model_training']==(label=='fixed') for e in train_events)
                assert all((e['active_dropout_modules']>0)==(label=='fixed') for e in train_events)
                assert any(g>0 for g in t['backward_gradient_norms'])
                run=out/f'{condition}_0M';ck=run/'model'
                model=GPT2LMHeadModel.from_pretrained(ck)
                weights[label]={k:v.clone() for k,v in model.state_dict().items()}
                tok=load_tokenizer(ck);heldout=[r['text'] for r in iter_jsonl(raw/'test.jsonl')]
                model.train();score_train=bits_per_byte(model,tok,heldout,None,32,1,'cpu')['bpb']
                model.eval();score_eval=bits_per_byte(model,tok,heldout,None,32,1,'cpu')['bpb']
                assert score_train==score_eval
                rows=[json.loads(line) for line in (run/'log.jsonl').read_text().splitlines()]
                final_key='final_eval_pruned' if condition=='E_progressive' else 'final_eval'
                encoding='old' if condition=='A_keep' else 'new'
                logged=next(row[final_key][encoding] for row in rows if final_key in row)
                assert abs(logged-score_eval)<1e-6
                traces[f'{condition}/{label}']=dict(source_commit=t['source_commit'],source_hashes=t['source_hashes'],
                    loaded_training=t['loads'][0]['training'],configured_dropout_rates=t['loads'][0]['dropout_rates'],
                    training_forwards=len(train_events),training_modes=sorted({e['model_training'] for e in train_events}),
                    active_dropout_counts=sorted({e['active_dropout_modules'] for e in train_events}),
                    evaluation_modes=sorted({e['model_training'] for e in eval_events}),
                    nonzero_backward_calls=sum(g>0 for g in t['backward_gradient_norms']),
                    reevaluation_same_checkpoint_max_bpb_difference=abs(score_train-score_eval),
                    saved_checkpoint_vs_logged_bpb_difference=abs(logged-score_eval))
                print(condition,label,'mode trace and saved-checkpoint evaluation verified',flush=True)
            key='transformer.h.0.attn.c_attn.weight'
            diff=float((weights['release'][key]-weights['fixed'][key]).abs().max())
            assert diff>0
            comparisons.append(dict(condition=condition,paired_dense_weight_max_abs_difference=diff))
        report=dict(scope='Synthetic tiny-model CPU diagnosis, not project model quality',seed=2026,
                    original_snapshot=args.old_root.name,traces=traces,paired_update_comparisons=comparisons,
                    conclusion='Original A/B/E update weights with dropout disabled; fixed trainer enables dropout. Same saved weights score identically from either entry mode; reevaluation cannot undo the changed updates.')
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
        print(json.dumps(comparisons,indent=2))


if __name__=='__main__':
    main()
