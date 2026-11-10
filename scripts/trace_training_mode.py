"""Passively trace an actual training CLI on toy data; never use on a live job."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import torch
from transformers import GPT2LMHeadModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--script', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('args', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    events, gradients, loads = [], [], []
    original = GPT2LMHeadModel.from_pretrained

    def loaded(cls, *pos, **kw):
        model = original(*pos, **kw)
        loads.append(dict(training=model.training, dropout_rates=sorted({m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)})))
        def before_forward(module, inputs):
            dropouts = [m for m in module.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
            events.append(dict(grad_enabled=torch.is_grad_enabled(), model_training=module.training,
                               configured_dropout_modules=len(dropouts), active_dropout_modules=sum(m.training for m in dropouts)))
        model.register_forward_pre_hook(before_forward)
        # This parameter is not replaced by vocabulary resizing or pruning.
        def grad_hook(grad):
            gradients.append(float(grad.detach().float().norm()))
        model.transformer.h[0].attn.c_attn.weight.register_hook(grad_hook)
        return model

    GPT2LMHeadModel.from_pretrained = classmethod(loaded)
    script = args.script.resolve()
    forwarded = args.args[1:] if args.args[:1] == ['--'] else args.args
    sys.argv = [str(script), *forwarded]
    try:
        runpy.run_path(str(script), run_name='__main__')
    finally:
        root = script.parents[1]
        report = dict(scope='passive toy CLI trace; not project model results',
                      source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                      source_hashes={rel:hashlib.sha256((root/rel).read_bytes()).hexdigest()
                                     for rel in ['scripts/train.py','src/toksw/metrics.py','src/toksw/model.py']},
                      loads=loads, training_forwards=[e for e in events if e['grad_enabled']],
                      evaluation_forwards=[e for e in events if not e['grad_enabled']],
                      backward_gradient_norms=gradients)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as f:
            f.write(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
