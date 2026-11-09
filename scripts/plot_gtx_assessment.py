"""Plot real diagnostic rates/memory; projections are not full training results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('report',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    report=json.loads(args.report.read_text())
    rows=[r for r in report['trials'] if r.get('status')=='passed' and r['precision']=='bf16' and r['micro_batch']==1]
    order={'A_keep':0,'B_swap_fvt':1,'E_progressive':2}
    rows.sort(key=lambda r:order[r['condition']])
    assert len(rows)==3 and {r['condition'] for r in rows}==set(order)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fields=['condition','estimated_training_tokens_per_second','projected_50M_train_hours','peak_allocated_bytes',
            'peak_reserved_bytes','checkpoint_bytes','checkpoint_save_seconds','micro_batch','grad_accum']
    with args.out.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    labels=['A / GPT-2','B / FVT 50k','E / union 84,717']
    fig,axes=plt.subplots(1,3,figsize=(12,4.2),layout='constrained')
    metrics=[('estimated_training_tokens_per_second','Measured microbatch-based\ntraining-rate estimate (tokens/s)',1),
             ('projected_50M_train_hours','50M training projection (hours)\nExcludes evaluation/setup/save',1),
             ('peak_allocated_bytes','Measured peak allocated\nGPU memory (GiB)',1/(1<<30))]
    for ax,(key,title,scale) in zip(axes,metrics):
        values=[r[key]*scale for r in rows]
        ax.bar(labels,values,color=['#3568a6','#d9842e','#3d9178'])
        for x,value in enumerate(values):
            ax.text(x,value+max(values)*.025,f'{value:.0f}' if key.startswith('estimated') else f'{value:.2f}',ha='center',fontsize=9)
        ax.set_title(title,fontsize=10)
        ax.tick_params(axis='x',rotation=20,labelsize=8)
        ax.set_ylim(0,max(values)*1.22)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    capacity=report['probe']['memory_bytes']/(1<<30)
    axes[2].axhline(capacity,color='#555555',linestyle='--',linewidth=1)
    axes[2].set_ylim(0,capacity*1.15)
    fig.suptitle(f"GTX 1060 feasibility — measured Slurm job {report['job_id']}")
    fig.supxlabel('BF16 (no native BF16 hardware), micro-batch 1 × accumulation 64; 65,536 tokens/step.\n'
                  'Four timed microbatches per condition. Diagnostic checkpoints removed. No model-quality claim.',fontsize=8)
    for ext in ['png','pdf']:
        fig.savefig(args.out.with_suffix('.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
