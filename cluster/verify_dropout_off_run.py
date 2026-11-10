"""Reevaluate immutable job-757238 checkpoints; publish PROVISIONAL pilot figures.

The training-mode regression cannot be repaired by reevaluation. This script
scores the learned dropout-disabled weights honestly, without retraining them.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import sys

TRAIN_COMMIT='31dbce7f645018eb221d55f5b7756872396de9b5'
CASES=[('A','A_keep_zh_50M','gpt2',50257,'old'),
       ('B','B_swap_fvt_zh_50M','wiki_zh70_gpt4_50k',50000,'new'),
       ('E','E_progressive_zh_50M','wiki_zh70_gpt4_50k',50000,'new')]
SCOPE='PROVISIONAL: eval-mode-trained / dropout disabled; Chinese 50M pilot'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def command(script,*args):
    subprocess.run([sys.executable,'-u','cluster/text_only.py',script,*map(str,args)],check=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--job-state',type=Path,required=True)
    ap.add_argument('--runs-root',type=Path,required=True)
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--tokenizers',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    from tokenizers import Tokenizer
    from run_chinese_gtx50 import quota
    state=json.loads(args.job_state.read_text())
    assert state['commit']==TRAIN_COMMIT and state['job_id']=='757238'
    saved={r['condition']:r for r in state['runs']}
    assert {'A_keep','B_swap_fvt','E_progressive'} <= set(saved), 'all three saved checkpoints required'
    storage=quota(0)  # Retain 1.1 GiB free; no weight or environment copies.
    args.out.mkdir(parents=True,exist_ok=False)
    report=dict(status='running',scope=SCOPE,provisional=True,training_job_id='757238',
                training_commit=TRAIN_COMMIT,training_mode='eval',effective_dropout='disabled',
                configured_dropout=.1,evaluation_mode='eval',evaluation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                quota=storage,cases=[])
    def write():
        (args.out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    write()
    try:
        test=args.raw/'wiki_zh/test.jsonl'
        stats=json.loads((args.raw/'wiki_zh/stats.json').read_text())
        raw_hash=sha(test);assert raw_hash==stats['audit']['test']['sha256']
        texts=[json.loads(line)['text'] for line in test.open()]
        assert len(texts)==1000
        n_bytes=sum(len(t.encode('utf-8')) for t in texts)
        assert n_bytes==stats['audit']['test']['utf8_bytes']
        log_paths=[];models=[]
        for letter,name,tok_name,vocab,encoding in CASES:
            run=args.runs_root/name; model=run/'model';weights=model/'model.safetensors'
            cfg=json.loads((run/'config.json').read_text());mcfg=json.loads((model/'config.json').read_text())
            assert cfg['commit']==TRAIN_COMMIT and cfg['micro_batch']==1 and cfg['grad_accum']==64
            assert cfg['eval_split']=='validation' and cfg['eval_batch']==1 and cfg['token_budget']==50_000_000
            assert mcfg['vocab_size']==vocab
            assert all(mcfg[k]==.1 for k in ['attn_pdrop','embd_pdrop','resid_pdrop'])
            record=saved[cfg['run_name'].replace('_zh','')]
            digest=sha(weights);assert digest==record['weights_sha256'] and weights.stat().st_size==record['weights_bytes']
            with weights.open('rb') as f:
                header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
            assert header['transformer.wte.weight']['shape']==[vocab,768]
            tok=Tokenizer.from_file(str(model/'tokenizer.json'))
            reference=Tokenizer.from_file(str(args.tokenizers/tok_name/'tokenizer.json'))
            assert tok.get_vocab()==reference.get_vocab() and tok.get_vocab_size()==vocab
            eos=tok.token_to_id('<|endoftext|>');assert eos is not None
            n_tokens=0
            for text in texts:
                ids=tok.encode(text,add_special_tokens=False).ids
                assert ids==reference.encode(text,add_special_tokens=False).ids
                assert tok.decode(ids,skip_special_tokens=False)==text
                n_tokens+=len(ids)+1
            rows=[json.loads(line) for line in (run/'log.jsonl').read_text().splitlines() if line]
            key='final_eval_pruned' if letter=='E' else 'final_eval'
            logged=next(r[key][encoding] for r in rows if key in r)
            assert math.isfinite(logged)
            dest=args.out/f'{letter}_bpb.json'
            command('scripts/eval_bpb.py',model,'--domain','wiki_zh','--raw',args.raw,'--batch-size','1','--out',dest)
            ev=json.loads(dest.read_text())
            assert ev['split']=='test' and ev['raw_sha256']==raw_hash
            assert ev['n_tokens']==n_tokens and ev['n_bytes']==n_bytes and not ev['model_training_after_eval']
            assert ev['batch_size']==1 and ev['block_size']==1024 and ev['device']=='cuda'
            assert math.isfinite(ev['bpb']) and ev['bpb']>0 and abs(ev['bpb']-logged)<1e-4
            assert abs(ev['nll_per_token']*ev['n_tokens']/math.log(2)/ev['n_bytes']-ev['bpb'])<1e-8
            ev.update(checkpoint=f'runs/{name}/model',scope=SCOPE,provisional=True,weights_sha256=digest)
            dest.write_text(json.dumps(ev,indent=2)+'\n')
            wall=next(r['wall_s'] for r in rows if 'wall_s' in r)
            log_dest=args.out/letter/'log.jsonl';log_dest.parent.mkdir()
            shutil.copyfile(run/'log.jsonl',log_dest)
            (log_dest.parent/'config.json').write_text(json.dumps({k:cfg[k] for k in
                ['block_size','micro_batch','grad_accum','eval_split']},indent=2)+'\n')
            log_paths.append(log_dest);models.append(model)
            report['cases'].append(dict(condition=letter,run=name,weights_sha256=digest,weights_bytes=weights.stat().st_size,
                tokenizer_sha256=sha(model/'tokenizer.json'),reference_tokenizer_semantics_verified=True,
                actual_tokens=record['actual_tokens'],test_bpb=ev['bpb'],logged_final_bpb=logged,
                trainer_wall_hours=wall/3600,eos_token_id=eos,n_test_documents=len(texts)))
            write()
        speed_path=args.out/'speed.csv'
        command('scripts/gen_speed.py',*models,'--domain','wiki_zh','--raw',args.raw,'--n-prompts','20',
                '--prompt-chars','200','--new-tokens','128','--out',speed_path)
        speeds=list(csv.DictReader(speed_path.open()));assert len(speeds)==3
        for row,case,model in zip(speeds,report['cases'],models):
            assert int(row['n_prompts'])==20 and int(row['prompt_chars'])==200 and int(row['new_tokens'])==128
            assert int(row['eos_token_id'])==case['eos_token_id']==int(row['pad_token_id'])
            assert row['raw_sha256']==raw_hash and row['tokenizer_sha256']==case['tokenizer_sha256']
            assert row['prompt_attention_mask']=='all_ones' and row['device']=='cuda'
            seconds=float(row['seconds']);chars=int(row['chars'])
            assert seconds>0 and chars>0 and math.isfinite(float(row['chars_per_second']))
            assert abs(chars/seconds-float(row['chars_per_second']))<1e-5
            row.update(checkpoint=f"runs/{case['run']}/model",scope=SCOPE,provisional=True)
            case['characters_per_second']=float(row['chars_per_second'])
            case['generation_tokens_per_second']=float(row['tokens_per_second'])
            assert sha(model/'model.safetensors')==case['weights_sha256'], 'checkpoint changed during verification'
        with speed_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(speeds[0]),lineterminator='\n');w.writeheader();w.writerows(speeds)
        comparison=[]
        for c in report['cases']:
            comparison.append(dict(condition=c['condition'],test_bpb=c['test_bpb'],characters_per_second=c['characters_per_second'],
                trainer_wall_hours=c['trainer_wall_hours'],training_scope=SCOPE,provisional=True))
        with (args.out/'comparison.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(comparison[0]),lineterminator='\n');w.writeheader();w.writerows(comparison)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(12,4.5),layout='constrained')
        for ax,key,title in zip(axes,['test_bpb','characters_per_second','trainer_wall_hours'],
                                  ['Test bits / UTF-8 byte (lower)','Generation characters / second (higher)','Reported trainer wall time (hours)']):
            values=[c[key] for c in comparison]
            ax.bar(['A / keep','B / FVT','E / pruned'],values,color=['#3568a6','#d9842e','#3d9178'])
            for i,value in enumerate(values):ax.text(i,value+max(values)*.025,f'{value:.3f}' if key=='test_bpb' else f'{value:.1f}',ha='center')
            ax.set_title(title,fontsize=9);ax.set_ylim(0,max(values)*1.18);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        fig.suptitle('PROVISIONAL — eval-mode-trained / dropout disabled\nChinese 50M pilot, GTX 1060')
        fig.supxlabel('20 MB tokenizer fits, not the proposed 1 GB study. Trainer wall_s includes periodic/final evaluation;\n'
                      'excludes initial evaluation, E pruning and checkpoint save. Speed: 20 shared prompts × 128 greedy tokens.',fontsize=8)
        for ext in ['png','pdf']:fig.savefig(args.out/f'comparison.{ext}',dpi=180)
        plt.close(fig)
        command('scripts/plot_results.py','--logs',*log_paths,'--out',args.out/'training',
                '--title','PROVISIONAL — eval-mode-trained / dropout disabled, Chinese 50M')
        (args.out/'README.md').write_text('# Provisional dropout-disabled Chinese 50M pilot\n\n'
            'These are verified inference measurements of checkpoints trained by release 31dbce7. '
            'Training used eval mode: gradients and updates occurred, but dropout was disabled. '
            'Reevaluation does not repair that training trajectory. Do not treat this as the intended '
            'dropout-0.1 study or pool it with corrected runs.\n\n'
            'All three weight hashes, vocabularies, reference-tokenizer behavior, full 1,000-document '
            'test coverage, log-to-checkpoint score agreement and speed protocol were checked. '
            'The scorer runs in eval mode; tokenizers were fitted on 20 MB, not 1 GB.\n')
        report.update(status='verified',test_raw_sha256=raw_hash,n_test_documents=1000,
                      caution='Valid metrics for dropout-disabled weights; provisional for intended training comparison.')
        write()
        files=[dict(path=str(p.relative_to(args.out)),bytes=p.stat().st_size,sha256=sha(p))
               for p in sorted(args.out.rglob('*')) if p.is_file()]
        assert all(f['bytes']<10*(1<<20) and Path(f['path']).suffix in {'.json','.jsonl','.csv','.png','.pdf','.md'} for f in files)
        (args.out/'manifest.json').write_text(json.dumps(dict(scope=SCOPE,provisional=True,files=files),indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    except Exception as e:
        report.update(status='failed',error=f'{type(e).__name__}: {e}');write();raise


if __name__=='__main__':main()
