"""Exactly fourteen fresh single updates; reuse immutable loaded source assets."""
import queue_train as q
from seven_synthesis import ARMS

def main():
    q.check_frozen(True)
    dg,_=q.runtime()
    with dg.r3.exclusive_lock():
        dg.r3.gpu_preflight()
        loaded=q.parent.load_inputs()
        q.engine.load_inputs=lambda:loaded
        for arm in ARMS:
            q.run(arm,True)
            print(arm+' full/tail calibration complete',flush=True)

if __name__=='__main__':main()
