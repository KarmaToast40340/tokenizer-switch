"""Exercise the GPU-check code path with tiny CPU fixtures, never project results."""
import importlib.util
from pathlib import Path

from transformers import GPT2Config, GPT2LMHeadModel

from toksw.data import encode_corpus
from toksw.utils import load_config


def test_smoke_uses_real_assets_and_prunes(tmp_path, corpus, old_tok, new_tok):
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('check_chinese_gpu', root / 'cluster/check_chinese_gpu.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    base = tmp_path / 'base'
    GPT2LMHeadModel(GPT2Config(vocab_size=old_tok.get_vocab_size(), n_positions=32,
                              n_embd=16, n_layer=1, n_head=2)).save_pretrained(base)
    for name, tok in [('gpt2', old_tok), ('wiki_zh70_gpt4_50k', new_tok)]:
        encode_corpus(tok, corpus, tmp_path / 'tokens/wiki_zh' / name)
    for config in ['A_keep', 'B_swap_fvt', 'E_progressive']:
        cfg = load_config(root / f'configs/train/{config}.yaml')
        cfg.update(base_model=str(base), tokens_dir=str(tmp_path / 'tokens'), domain='wiki_zh',
                   new_name='wiki_zh70_gpt4_50k', block_size=32, micro_batch=1, grad_accum=2)
        report = mod.check_condition(cfg, old_tok, new_tok, ['validation only 中文'], device='cpu')
        assert report['status'] == 'passed' and report['finite_losses'] == 4
        assert report['final_vocab_size'] == (old_tok if config == 'A_keep' else new_tok).get_vocab_size()
        if config == 'E_progressive':
            assert report['vocab_size'] > report['final_vocab_size']
