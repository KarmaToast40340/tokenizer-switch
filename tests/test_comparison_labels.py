"""Figures must retain their training protocol, independent of panel titles."""
import importlib.util
from pathlib import Path
import pytest


@pytest.mark.parametrize('title',[
    'PROVISIONAL — eval-mode-trained / dropout disabled / Chinese 50M pilot',
    'Corrected Chinese 50M — dropout 0.1 / 20 MB tokenizer pilots'])
def test_comparison_scope_survives_axis_titles(tmp_path,monkeypatch,title):
    spec=importlib.util.spec_from_file_location('comparison_labels',Path(__file__).resolve().parents[1]/'cluster/verify_dropout_off_run.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=plt.close;seen=[]
    def closed(fig=None):
        if hasattr(fig,'_suptitle') and fig._suptitle:
            seen.append(fig._suptitle.get_text())
        original(fig)
    monkeypatch.setattr(plt,'close',closed)
    # Synthetic figure fixture only; files remain in pytest's temporary folder.
    rows=[dict(test_bpb=2.,characters_per_second=100.,trainer_wall_hours=10.)]*3
    mod.comparison_figure(rows,tmp_path,title)
    assert seen==[title+'\nGTX 1060']
    assert (tmp_path/'comparison.png').is_file() and (tmp_path/'comparison.pdf').is_file()
