"""Fail rather than silently train or return from evaluation in the wrong mode."""
import torch


def require_mode(model, training, expected_dropout=None):
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    if model.training != training or any(m.training != training for m in dropouts):
        raise RuntimeError(f'model/dropout mode mismatch; expected training={training}')
    if expected_dropout is not None and (not dropouts or any(m.p != expected_dropout for m in dropouts)):
        raise RuntimeError(f'configured dropout does not match {expected_dropout}')
