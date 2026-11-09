"""Run a text-model CLI without DL_HW's optional OpenCV/vision imports.

Changes discovery flags in this process only; shared packages remain untouched.
Validated with transformers 4.49.0 on the course compute image.
"""
import runpy
import sys

import transformers.utils.import_utils as imports

imports._cv2_available = False
imports._torchvision_available = False
if len(sys.argv) < 2:
    raise SystemExit('usage: text_only.py SCRIPT [ARGS...]')
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
