"""Offline integration check of existing A/B/C/E training/pruning on toy data.

These random tiny models are test fixtures, never project results.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from transformers import GPT2Config, GPT2LMHeadModel
from toksw.data import encode_corpus


def test_all_conditions_with_validation_and_pruning(tmp_path, corpus, old_tok, new_tok):
    root = Path(__file__).resolve().parents[1]
    base = tmp_path / "base"
    GPT2LMHeadModel(GPT2Config(vocab_size=old_tok.get_vocab_size(),n_positions=32,n_embd=16,n_layer=1,n_head=2)).save_pretrained(base)
    old, new = tmp_path/"old.json", tmp_path/"new.json"
    old_tok.save(str(old)); new_tok.save(str(new))
    raw = tmp_path/"raw/code"; raw.mkdir(parents=True)
    for split, text in [("validation","validation only 中文"),("test","held-out test only")]:
        (raw/f"{split}.jsonl").write_text(json.dumps(dict(text=text))+"\n")
    for name,tok in [("old",old_tok),("new",new_tok)]:
        encode_corpus(tok,corpus,tmp_path/"tokens/code"/name)
    env = os.environ | dict(PYTHONPATH=str(root/"src"),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",OMP_NUM_THREADS="1",TOKENIZERS_PARALLELISM="false")
    for condition in ["A_keep","B_swap_fvt","C_swap_random","E_progressive"]:
        trace=tmp_path/f"{condition}_mode_trace.json"
        command=[sys.executable,"scripts/trace_training_mode.py","--script","scripts/train.py","--out",str(trace),"--",f"configs/train/{condition}.yaml",
                 f"base_model={base}",f"old_tokenizer={old}",f"new_tokenizer={new}","old_name=old","new_name=new",
                 f"raw_dir={tmp_path/'raw'}",f"tokens_dir={tmp_path/'tokens'}",f"out={tmp_path/'runs'}",
                 f"run_name={condition}","token_budget=128","block_size=32","micro_batch=1","grad_accum=1",
                 "eval_every=2","eval_docs=1","eval_batch=1","eval_split=validation","log_every=1"]
        subprocess.run(command,cwd=root,env=env,check=True,capture_output=True,text=True,timeout=180)
        modes=json.loads(trace.read_text())
        assert modes["training_forwards"] and all(e["model_training"] and e["active_dropout_modules"]>0 for e in modes["training_forwards"])
        assert modes["evaluation_forwards"] and all(not e["model_training"] and not e["active_dropout_modules"] for e in modes["evaluation_forwards"])
        assert any(g>0 for g in modes["backward_gradient_norms"])
        run=tmp_path/"runs"/f"{condition}_0M"
        rows=[json.loads(line) for line in (run/"log.jsonl").read_text().splitlines()]
        assert any(r.get("step")==2 and "eval" in r for r in rows)
        final=next(r["final_eval"] for r in rows if "final_eval" in r)
        final_row=next(r for r in rows if "final_eval" in r)
        assert final_row["training_mode_checks"]==4 and final_row["model_training_after_eval"]
        assert all(r["model_training"] for r in rows if "train_bpb" in r)
        assert all(r["model_training_after_eval"] for r in rows if "eval" in r)
        if condition=="A_keep":
            before=(run/"model/model.safetensors").read_bytes()
            retry=subprocess.run([sys.executable,"scripts/train.py",*command[command.index("--")+1:]],
                cwd=root,env=env,capture_output=True,text=True,timeout=180)
            assert retry.returncode and "FileExistsError" in retry.stderr
            assert (run/"model/model.safetensors").read_bytes()==before
        cfg=json.loads((run/"model/config.json").read_text())
        assert cfg["vocab_size"] == (old_tok.get_vocab_size() if condition=="A_keep" else new_tok.get_vocab_size())
        if condition=="E_progressive":
            pruned=next(r["final_eval_pruned"] for r in rows if "final_eval_pruned" in r)
            assert pruned["new"] <= final["new"] + 1e-5
