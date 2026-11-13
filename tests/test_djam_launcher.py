"""Execute the Slurm wrapper with a stub interpreter; never submit a GPU job."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from toksw.utils import load_config


@pytest.mark.parametrize('condition',['A_keep','B_swap_fvt','C_swap_random','E_progressive'])
def test_uses_existing_interpreter_and_corrected_matched_settings(tmp_path,condition):
    root=Path(__file__).resolve().parents[1]
    capture=tmp_path/'argv.json'
    python=tmp_path/'existing-python'
    python.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
        'if sys.argv[1]!="-c":\n'
        ' open(os.environ["ARGV_CAPTURE"],"w").write(json.dumps(sys.argv[1:]))\n')
    python.chmod(0o700)
    base=tmp_path/'base';base.mkdir();(base/'model.safetensors').touch()
    env=os.environ|dict(DJAM_PYTHON=str(python),DJAM_BASE_MODEL=str(base),SLURM_SUBMIT_DIR=str(root),
        ARGV_CAPTURE=str(capture),USER='djam-test',VENV='/does-not-exist/do-not-activate')
    subprocess.run(['bash','cluster/djam_code50.sbatch',condition],cwd=root,env=env,
        check=True,capture_output=True,text=True,timeout=30)
    args=json.loads(capture.read_text())
    assert args[:3]==['-u','cluster/text_only.py','scripts/train.py']
    cfg=load_config(root/args[3],args[4:])
    assert cfg['base_model']==str(base) and cfg['domain']=='code'
    assert cfg['old_tokenizer']=='artifacts/tokenizers/gpt2'
    assert cfg['new_tokenizer']=='artifacts/tokenizers/code70_gpt4_32k'
    assert cfg['expected_dropout']==.1 and cfg['eval_split']=='validation'
    assert cfg['micro_batch']==1 and cfg['grad_accum']==64 and cfg['eval_batch']==1
    assert cfg['token_budget']//65536*65536==49_938_432
    assert cfg['out']=='runs/djam_code50_corrected' and cfg['account']=='djam-test'
    assert cfg['run_name']==condition+'_code_dropout01'
