"""Check FVT / random init, progressive union, pruning and the mixed sampler on real assets (CPU).

    PYTHONPATH=src python scripts/check_transfer.py --base <gpt2 snapshot dir> \
        --old artifacts/tokenizers/gpt2 --new artifacts/tokenizers/code70_gpt4_32k \
        --eval data/code200/raw_pinned/code/validation.jsonl \
        --tokens data/code50_pkg/code50-djam-2026-11-13/data/tokens/code \
        --out results/transfer-check-code.json
"""
import argparse
import gc
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel

from toksw import EOT, model as M
from toksw.data import MixedBatchSampler, TokenFile, iter_jsonl, token_byte_lengths
from toksw.metrics import bits_per_byte
from toksw.schedule import p_new
from toksw.tok import eot_id, load_tokenizer
from toksw.vocab import _id_to_token, allowed_rows, build_union, fvt_pieces, prune_to_new

CHECKS = []


def check(name, ok, **numbers):
    CHECKS.append(dict(check=name, status="PASS" if ok else "FAIL", **numbers))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {json.dumps(numbers, default=str)}", flush=True)
    return ok


def tied(model):
    return model.lm_head.weight.data_ptr() == model.transformer.wte.weight.data_ptr()


def bpb(model, tok, texts, id_map, args, allowed=None):
    t = time.time()
    r = bits_per_byte(model, tok, texts, id_map, args.block_size, args.batch, "cpu", allowed)
    r["seconds"] = round(time.time() - t, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--tokens", required=True, help="directory holding gpt2 and new encodings")
    ap.add_argument("--new-name", default="code70_gpt4_32k")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--total-steps", type=int, default=762)
    ap.add_argument("--rho", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(0)
    t_start = time.time()

    old_tok, new_tok = load_tokenizer(args.old), load_tokenizer(args.new)
    old_tokens, new_tokens = _id_to_token(old_tok), _id_to_token(new_tok)
    texts = [d["text"] for d in iter_jsonl(args.eval, args.n_docs)]
    info = dict(n_eval_docs=len(texts), eval_bytes=sum(len(t.encode()) for t in texts))

    # 1. Union vocabulary
    u = build_union(old_tok, new_tok)
    eot_new = eot_id(new_tok)
    check("1 union", u.n_old == len(old_tokens) == 50257
          and u.size == u.n_old + u.n_new_only
          and int(u.new2union[eot_new]) == 50256 == eot_id(old_tok)
          and np.array_equal(u.old2union, np.arange(u.n_old))
          and len(set(u.new2union.tolist())) == len(new_tokens)
          and all(u.tokens[int(u.new2union[j])] == s for j, s in enumerate(new_tokens)),
          n_old=u.n_old, n_new=len(new_tokens), n_new_only=u.n_new_only, size=u.size,
          n_shared=len(new_tokens) - u.n_new_only, new_eot_id=eot_new,
          new_eot_union_id=int(u.new2union[eot_new]))

    # 2. FVT pieces spell every new token
    old_vocab = old_tok.get_vocab(with_added_tokens=True)
    fails, lens = [], []
    for s in new_tokens:
        p = fvt_pieces(old_tok, s)
        if "".join(old_tokens[i] for i in p) != s:
            fails.append(s)
        if s not in old_vocab:
            lens.append(len(p))
    lens = np.array(lens)
    examples = {}
    for s in ["ĠĠĠ", "ĠĠĠĠ", "ĠĠĠĠĠĠĠ", "ĠĠĠĠĠĠĠĠ", "ĊĠĠĠĠĠĠĠ", "ĊĠĠĠĠĠĠĠĠĠĠĠ", "Ġreturn", ".append",
              "Ġself", "):Ċ", "Ġ__init__"]:
        if s in new_tok.get_vocab():
            examples[s] = dict(in_gpt2=s in old_vocab, pieces=[old_tokens[i] for i in fvt_pieces(old_tok, s)])
    longest = [new_tokens[j] for j in np.argsort([-len(fvt_pieces(old_tok, s)) if s not in old_vocab else 0
                                                 for s in new_tokens])[:3]]
    for s in longest:
        examples[s] = dict(in_gpt2=False, pieces=[old_tokens[i] for i in fvt_pieces(old_tok, s)])
    check("2 fvt pieces", not fails, n_failures=len(fails), failures=fails[:5],
          new_only_pieces_mean=round(float(lens.mean()), 3), new_only_pieces_max=int(lens.max()),
          new_only_single_piece=int((lens == 1).sum()), examples=examples)

    # 3-4. Build every condition, check init rows, tying and step-0 bpb
    res = {}
    gpt2 = GPT2LMHeadModel.from_pretrained(args.base)
    old_emb = gpt2.transformer.wte.weight.data.float().clone()
    del gpt2
    shared_new = [j for j, s in enumerate(new_tokens) if s in old_vocab]
    only_new = [j for j, s in enumerate(new_tokens) if s not in old_vocab]
    rng = np.random.default_rng(0)
    sample = rng.choice(only_new, size=min(100, len(only_new)), replace=False)
    shared_old_idx = torch.tensor([old_vocab[new_tokens[j]] for j in shared_new])

    A = M.build("keep", old_tok, None, args.base)
    check("7 tied A", tied(A.model))
    res["A"] = bpb(A.model, old_tok, texts, A.old_map, args)
    del A; gc.collect()

    B = M.build("swap", old_tok, new_tok, args.base, "fvt", 0)
    embB = B.model.transformer.wte.weight.data
    check("7 tied B", tied(B.model))
    fvt_err = max(float((embB[j] - old_emb[fvt_pieces(old_tok, new_tokens[j])].mean(0)).abs().max()) for j in sample)
    check("3 init B", torch.equal(embB[shared_new], old_emb[shared_old_idx]) and fvt_err < 1e-6,
          shared_rows_exact=bool(torch.equal(embB[shared_new], old_emb[shared_old_idx])),
          fvt_rows_sampled=len(sample), fvt_max_abs_err=fvt_err)
    res["B"] = bpb(B.model, new_tok, texts, B.new_map, args)
    embB_only = embB[only_new].clone()
    del B, embB; gc.collect()

    C = M.build("swap", old_tok, new_tok, args.base, "random", 0)
    embC = C.model.transformer.wte.weight.data
    check("7 tied C", tied(C.model))
    diff = (embC[only_new] - embB_only).norm(dim=1) / embB_only.norm(dim=1)
    check("3 init C", torch.equal(embC[shared_new], old_emb[shared_old_idx]) and float(diff.min()) > 0.1,
          shared_rows_exact=bool(torch.equal(embC[shared_new], old_emb[shared_old_idx])),
          random_vs_fvt_rel_diff_min=float(diff.min()), random_vs_fvt_rel_diff_mean=float(diff.mean()))
    res["C"] = bpb(C.model, new_tok, texts, C.new_map, args)
    del C, embC; gc.collect()

    E = M.build("progressive", old_tok, new_tok, args.base, "fvt", 0)
    embE = E.model.transformer.wte.weight.data
    check("7 tied E", tied(E.model))
    e_only = torch.as_tensor(E.union.new2union[only_new])
    check("3 init E", torch.equal(embE[:u.n_old], old_emb) and torch.equal(embE[e_only], embB_only)
          and E.model.transformer.wte.weight.shape[0] == u.size,
          old_rows_exact=bool(torch.equal(embE[:u.n_old], old_emb)),
          new_only_rows_equal_B=bool(torch.equal(embE[e_only], embB_only)), rows=u.size)
    res["E_old"] = bpb(E.model, old_tok, texts, E.old_map, args)
    res["E_new"] = bpb(E.model, new_tok, texts, E.new_map, args)
    # opt-in mask_foreign_logits: each sequence's softmax restricted to its tokenizer's rows
    old_allowed, new_allowed = allowed_rows(E.union)
    res["E_old_masked"] = bpb(E.model, old_tok, texts, E.old_map, args, old_allowed)
    res["E_new_masked"] = bpb(E.model, new_tok, texts, E.new_map, args, new_allowed)
    for k, v in res.items():
        print(f"  step0 {k}: bpb={v['bpb']:.5f} tokens={v['n_tokens']} ({v['seconds']}s)", flush=True)
    b = {k: v["bpb"] for k, v in res.items()}
    print(json.dumps({k: round(v, 6) for k, v in b.items()}))
    check("4a step0 B < C", b["B"] < b["C"] - 0.1, B=b["B"], C=b["C"], C_minus_B=b["C"] - b["B"])
    check("4b step0 E_new ~ B (unmasked)", 0 <= b["E_new"] - b["B"] < 0.05, E_new_minus_B=b["E_new"] - b["B"])
    check("4c step0 E_old ~ A (unmasked)", 0 <= b["E_old"] - b["A"] < 0.1, E_old_minus_A=b["E_old"] - b["A"])
    check("4d step0 E_old masked == A", abs(b["E_old_masked"] - b["A"]) < 1e-4,
          E_old_masked=b["E_old_masked"], A=b["A"], abs_diff=abs(b["E_old_masked"] - b["A"]))
    check("4e step0 E_new masked == B", abs(b["E_new_masked"] - b["B"]) < 1e-4,
          E_new_masked=b["E_new_masked"], B=b["B"], abs_diff=abs(b["E_new_masked"] - b["B"]))

    # 5. Pruning exactly like scripts/train.py
    model = E.model
    model.eval()
    eot = eot_id(new_tok)
    ids = [eot] + [i for e in new_tok.encode_batch(texts[:4], add_special_tokens=False) for i in e.ids + [eot]]
    x_new = torch.tensor(ids[: args.block_size])[None]
    with torch.no_grad():
        union_logits = model(input_ids=torch.as_tensor(E.new_map)[x_new]).logits[0][:, torch.as_tensor(u.new2union)]
    pruned = prune_to_new(model.transformer.wte.weight.data, E.union)
    M._set_embeddings(model, pruned)
    P = M.Setup(model, None, np.arange(pruned.shape[0]), E.token_bytes[E.union.new2union])
    check("7 tied pruned", tied(model) and model.transformer.wte.weight.shape[0] == len(new_tokens),
          rows=model.transformer.wte.weight.shape[0])
    with torch.no_grad():
        pruned_logits = model(input_ids=x_new).logits[0]
        lsm_diff = float((torch.log_softmax(pruned_logits, -1) - torch.log_softmax(union_logits, -1)).abs().max())
    logit_diff = float((pruned_logits - union_logits).abs().max())
    argmax_agree = float((pruned_logits.argmax(-1) == union_logits.argmax(-1)).float().mean())
    res["pruned"] = bpb(model, new_tok, texts, P.new_map, args)
    gap = res["E_new"]["bpb"] - res["pruned"]["bpb"]
    check("5 pruning", 0 <= gap < 0.05 and logit_diff < 1e-3 and argmax_agree == 1.0,
          E_new=round(res["E_new"]["bpb"], 6), pruned=round(res["pruned"]["bpb"], 6), E_new_minus_pruned=gap,
          max_abs_logit_diff=logit_diff, max_abs_logsoftmax_diff_after_renorm=lsm_diff,
          argmax_agreement=argmax_agree, tokens_compared=x_new.shape[1])
    with tempfile.TemporaryDirectory() as d:
        model.save_pretrained(d)
        reloaded = GPT2LMHeadModel.from_pretrained(d)
    check("7 tied reloaded", tied(reloaded))
    res["reloaded"] = bpb(reloaded, new_tok, texts, P.new_map, args)
    rd = abs(res["reloaded"]["bpb"] - res["pruned"]["bpb"])
    check("5 save/reload", rd < 1e-5, pruned=res["pruned"]["bpb"], reloaded=res["reloaded"]["bpb"], abs_diff=rd)
    del reloaded, model, E; gc.collect()

    # 6. Progressive schedule with the real encodings
    tokens = Path(args.tokens)
    old_f, new_f = TokenFile(tokens / "gpt2"), TokenFile(tokens / args.new_name)
    sampler = MixedBatchSampler(old_f, new_f, u.old2union, u.new2union, token_byte_lengths(u.tokens, {EOT}),
                                lambda s: p_new(s, args.total_steps, args.rho), args.block_size, 8, 1234)
    sched, ok6 = [], True
    for step in (0, 190, 381, 600):
        fr = [sampler.get_batch(step, m)[2]["frac_new"] for m in range(20)]
        p = p_new(step, args.total_steps, args.rho)
        sched.append(dict(step=step, p_new=round(p, 4), frac_new=round(float(np.mean(fr)), 4)))
        ok6 &= abs(np.mean(fr) - p) < 0.2
    mismatches, compared = 0, 0
    for step in (0, 190, 381, 600):
        for micro in range(3):
            r = np.random.default_rng([1234, step, micro])
            for d in r.integers(0, old_f.n_docs, size=8):
                a = old_tok.decode(old_f.window(int(d), 120).tolist(), skip_special_tokens=False)[:200]
                c = new_tok.decode(new_f.window(int(d), 120).tolist(), skip_special_tokens=False)[:200]
                n = min(len(a), len(c), 200) - 4  # last chars may be a cut multi-byte token
                compared += 1
                mismatches += a[:n] != c[:n]
    x, _, inf = sampler.get_batch(600, 0)
    union_ids_ok = int(x.max()) < u.size
    check("6 progressive sampler", ok6 and mismatches == 0 and union_ids_ok, schedule=sched,
          prefix_pairs_compared=compared, prefix_mismatches=mismatches, n_docs=old_f.n_docs,
          step600_frac_new=inf["frac_new"], max_model_id=int(x.max()))

    out = dict(info=info, base=args.base, old=args.old, new=args.new, eval=args.eval,
               bpb={k: v for k, v in res.items()}, checks=CHECKS,
               all_pass=all(c["status"] == "PASS" for c in CHECKS), seconds=round(time.time() - t_start, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str) + "\n")
    print(f"all_pass={out['all_pass']} ({out['seconds']}s) -> {args.out}")


if __name__ == "__main__":
    main()
