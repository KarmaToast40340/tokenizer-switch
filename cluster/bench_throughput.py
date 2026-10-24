"""Mesure le debit d'entrainement de GPT-2 small (124M) sur le GPU du cluster.
Usage : python bench_throughput.py --micro 8 --seq 1024 --steps 60 [--compile]
Affiche les tokens/s apres echauffement, et la memoire max utilisee."""
import argparse, time, torch
from transformers import GPT2Config, GPT2LMHeadModel

p = argparse.ArgumentParser()
p.add_argument("--micro", type=int, default=8)
p.add_argument("--seq", type=int, default=1024)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--warmup", type=int, default=10)
p.add_argument("--vocab", type=int, default=50257)
p.add_argument("--compile", action="store_true")
a = p.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
dev = "cuda"
cfg = GPT2Config(vocab_size=a.vocab, n_positions=a.seq, attn_implementation="sdpa")
model = GPT2LMHeadModel(cfg).to(dev)
if a.compile:
    model = torch.compile(model)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
x = torch.randint(0, a.vocab, (a.micro, a.seq), device=dev)

def step():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=x, labels=x).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

for _ in range(a.warmup):
    step()
torch.cuda.synchronize()
t = time.time()
for _ in range(a.steps):
    step()
torch.cuda.synchronize()
dt = time.time() - t
tps = a.steps * a.micro * a.seq / dt
print(f"GPU {torch.cuda.get_device_name()} | micro {a.micro} seq {a.seq} vocab {a.vocab} compile {a.compile}")
print(f"{tps:,.0f} tokens/s | 50M tokens = {50e6/tps/3600:.1f} h | 200M tokens = {200e6/tps/3600:.1f} h")
print(f"memoire max {torch.cuda.max_memory_allocated()/2**30:.2f} Go")
