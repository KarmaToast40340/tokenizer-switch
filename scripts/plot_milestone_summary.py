"""Presentation figure from frozen held-out measurements and verified assets."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    rows = []
    for profile, domain in [('milestone', 'code'), ('chinese', 'wiki_zh')]:
        selected = json.loads(Path(f'results/{profile}/selection.json').read_text())['selected']['tokenizer']
        measured = list(csv.DictReader(Path(f'results/{profile}/test.csv').open()))
        rows.append(next(r for r in measured if r['tokenizer'] == selected))
    manifest = json.loads(Path('results/assets-manifest.json').read_text())
    capacities = []
    for profile in ['code50', 'chinese']:
        for item in manifest['files']:
            if item['profile'] == profile and '/tokens/' in item['source'] and item['source'].endswith('.json'):
                meta = json.loads(Path(item['source']).read_text())
                capacities.append((('Code' if profile == 'code50' else 'Chinese') +
                                  (' / GPT-2' if Path(item['source']).stem == 'gpt2' else ' / selected'),
                                   meta['n_tokens'] / 1e6))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), layout='constrained')
    for x, row, target in zip(range(2), rows, ['code', 'wiki_zh']):
        for offset, domain, color, label in [(-.18, target, '#3568a6', 'Target language'),
                                             (.18, 'wiki_en', '#d9842e', 'English')]:
            value = float(row[f'nsl_{domain}'])
            axes[0].bar(x+offset, value, width=.34, color=color, label=label if x == 0 else None)
            axes[0].text(x+offset, value+.025, f'{value:.4f}', ha='center', fontsize=9)
    axes[0].axhline(1, color='#444444', linestyle='--', linewidth=1, label='GPT-2 baseline')
    axes[0].set(xticks=[0, 1], xticklabels=['Code (32k)', 'Chinese (50k)'],
                ylim=(0, 1.3), ylabel='Test NSL vs GPT-2 (lower is better)', title='Measured compression')
    axes[0].legend(fontsize=8, loc='lower right')
    labels, values = zip(*capacities)
    axes[1].barh(labels, values, color=['#3568a6', '#3568a6', '#3d9178', '#3d9178'])
    for y, value in enumerate(values):
        axes[1].text(value+3, y, f'{value:.1f}M', va='center', fontsize=9)
    for budget in [50, 200]:
        axes[1].axvline(budget, color='#555555', linestyle='--', linewidth=1)
        axes[1].text(budget+2, 1.65, f'{budget}M', rotation=90, fontsize=8)
    axes[1].set(xlim=(0, 330), xlabel='Encoded training tokens (millions)', title='50M-ready model corpora')
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis='y' if ax is axes[0] else 'x', alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Frozen pilot tokenizers: compression and training readiness', fontsize=13)
    fig.supxlabel('Tokenizer fits: 20 MB per variant (pilot). Model corpora: code 250 MB; Chinese 400 MB.\n'
                  'Capacity is not unique coverage. Dropout-disabled model results are separate; corrected comparisons pending.', fontsize=9)
    out = Path('results/milestone_summary')
    for ext in ['png', 'pdf']:
        fig.savefig(out.with_suffix('.'+ext), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
