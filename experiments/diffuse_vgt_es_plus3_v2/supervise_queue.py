"""Run bounded formal queue then frozen development; fail closed, no resume."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def main():
    state=ROOT/'supervisor_state.json'
    def write(**value):state.write_text(json.dumps(dict(updated=time.time(),**value),indent=2)+'\n')
    try:
        for mode in ('queue','dev'):
            write(status='RUNNING',stage=mode)
            with (ROOT/f'supervisor_{mode}.log').open('x') as log:
                subprocess.run([sys.executable,str(ROOT/'queue_train.py'),mode],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=43200)
        write(status='COMPLETE',stage='development_frozen_real_scoring_pending')
    except Exception as exc:
        write(status='FAILED',error=repr(exc))
        raise

if __name__=='__main__':main()
