"""Exercise saved-checkpoint evaluation offline using a tiny random test model."""
import json
import os
from pathlib import Path
import subprocess
import sys

from transformers import GPT2Config, GPT2LMHeadModel


def test_generation_suppresses_actual_tokenizer_eot(tmp_path, old_tok):
    """A swapped vocab can have EOT=0 while config still names GPT-2's 50256.

    Zero logits would greedily choose EOT every step without correct suppression,
    yielding zero decoded characters despite the fixed-new-token protocol.
    """
    import csv
    import torch
    from toksw.tok import eot_id
    assert eot_id(old_tok) == 0
    root = Path(__file__).resolve().parents[1]
    ckpt = tmp_path / 'model'
    model = GPT2LMHeadModel(GPT2Config(vocab_size=old_tok.get_vocab_size(), n_positions=64,
                                     n_embd=16, n_layer=1, n_head=2, eos_token_id=50256))
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    model.save_pretrained(ckpt)
    old_tok.save(str(ckpt / 'tokenizer.json'))
    raw = tmp_path / 'raw/code'; raw.mkdir(parents=True)
    (raw / 'test.jsonl').write_text(json.dumps(dict(text='def f(): return 42'))+'\n')
    out = tmp_path / 'speed.csv'
    env = os.environ | dict(PYTHONPATH=str(root/'src'), HF_HUB_OFFLINE='1', OMP_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false')
    subprocess.run([sys.executable,'scripts/gen_speed.py',str(ckpt),'--domain','code','--raw',str(raw.parent),
                    '--n-prompts','1','--prompt-chars','8','--new-tokens','4','--out',str(out)],
                   cwd=root,env=env,check=True,capture_output=True,text=True,timeout=180)
    row = next(csv.DictReader(out.open()))
    assert int(row['chars']) > 0
    assert row['eos_token_id'] == row['pad_token_id'] == '0'
    assert row['prompt_attention_mask'] == 'all_ones'
    assert float(row['seconds']) > 0
    from toksw.splits import file_sha256
    assert row['raw_sha256'] == file_sha256(raw/'test.jsonl')
    assert row['tokenizer_sha256'] == file_sha256(ckpt/'tokenizer.json')
    bpb = tmp_path/'bpb.json'
    subprocess.run([sys.executable,'scripts/eval_bpb.py',str(ckpt),'--domain','code','--raw',str(raw.parent),
                    '--block-size','32','--batch-size','1','--out',str(bpb)],
                   cwd=root,env=env,check=True,capture_output=True,text=True,timeout=180)
    evaluation = json.loads(bpb.read_text())
    assert evaluation['model_training_after_eval'] is False
    assert evaluation['n_bytes'] == len('def f(): return 42'.encode())
    assert evaluation['raw_sha256'] == row['raw_sha256']
    assert evaluation['tokenizer_sha256'] == row['tokenizer_sha256']


def test_saved_checkpoint_bpb_and_speed_cli(tmp_path, old_tok):
    root = Path(__file__).resolve().parents[1]
    ckpt = tmp_path / "toy_test_only" / "model"
    ckpt.mkdir(parents=True)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=old_tok.get_vocab_size(), n_positions=64,
                                     n_embd=16, n_layer=1, n_head=2))
    model.save_pretrained(ckpt)
    old_tok.save(str(ckpt / "tokenizer.json"))
    raw = tmp_path / "raw" / "code"
    raw.mkdir(parents=True)
    (raw / "test.jsonl").write_text(json.dumps(dict(text="def f():\n    return 42\n")) + "\n")
    out = tmp_path / "evaluation"
    env = os.environ | dict(PYTHONPATH=str(root / "src"), HF_HUB_OFFLINE="1", OMP_NUM_THREADS="1",
                            TOKENIZERS_PARALLELISM="false")
    subprocess.run([sys.executable, "scripts/evaluate_checkpoints.py", str(ckpt),
                    "--raw", str(raw.parent), "--out", str(out), "--block-size", "32",
                    "--speed", "--prompt-chars", "8", "--new-tokens", "4"],
                   cwd=root, env=env, check=True, capture_output=True, text=True, timeout=180)
    result = json.loads((out / "bpb.json").read_text())[0]
    assert result["n_bytes"] == len("def f():\n    return 42\n".encode())
    assert result["bpb"] > 0
    assert "chars_per_second" in (out / "speed.csv").read_text()
