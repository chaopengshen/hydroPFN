"""Finite two-seed decoder comparison, one seed's complete matrix per GPU."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_recovery_stage1 import stamp, verify_sources
from run_recovery_stage2 import free_gpu

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'logs/decoder_comparison'


def status(tag, payload):
    p=OUT/f'{tag}.json.tmp'
    p.write_text(json.dumps(payload,indent=2))
    p.replace(OUT/f'{tag}.json')


def execute(gpu,tag,args):
    deadline=time.monotonic()+86400
    while not free_gpu(gpu):
        status(tag,dict(state='waiting_for_gpu',gpu=gpu,updated=stamp()))
        if time.monotonic()>deadline:
            raise RuntimeError(f'GPU {gpu} unavailable after 24 hours')
        time.sleep(30)
    digest=verify_sources()
    command=[sys.executable,'-u',str(ROOT/'experiments/camels531_decoders.py'),*args]
    env=os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),
               PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
               HYDROPFN_CAMELS_ROOT='/nfs/data/cxs1024/hydroPFN/data')
    rec=dict(state='running',gpu=gpu,command=command,started=stamp(),source_manifest_sha256=digest)
    with (OUT/f'{tag}.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        rec['pid']=proc.pid;status(tag,rec)
        print(f'{stamp()} started {tag} GPU={gpu} PID={proc.pid}',flush=True)
        code=proc.wait()
    rec.update(state='complete' if code==0 else 'failed',exit_code=code,finished=stamp())
    status(tag,rec)
    if code:
        raise RuntimeError(f'{tag} failed; inspect its log')
    print(f'{stamp()} complete {tag}',flush=True)


def worker(gpu,seed,a,cache):
    for decoder,inputs in [('lstm','raw'),('tcn','raw'),('lstm','hybrid'),
                           ('tcn','hybrid'),('pointwise','hybrid')]:
        tag=f'decoder_{decoder}_{inputs}_s{seed}'
        execute(gpu,tag,['train','--cache',str(cache),'--tag',tag,'--decoder',decoder,
                        '--inputs',inputs,'--seed',str(seed),'--epochs',str(a.epochs),
                        '--sampling','fixed'])


def main(a):
    OUT.mkdir(parents=True,exist_ok=True)
    verify_sources()
    if a.detach:
        with (OUT/'supervisor.log').open('x') as log:
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--gpus',a.gpus,
                    '--source',a.source,'--epochs',str(a.epochs)],cwd=ROOT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (OUT/'supervisor_pid.txt').write_text(str(proc.pid))
        print(json.dumps(dict(pid=proc.pid,logs=str(OUT))))
        return
    gpus=[int(v) for v in a.gpus.split(',')]
    if len(gpus)!=2 or len(set(gpus))!=2:
        raise ValueError('two distinct GPUs required')
    cache=OUT/'cache'
    for seed,gpu in enumerate(gpus):
        for decoder,inputs in [('lstm','raw'),('tcn','raw'),('lstm','hybrid'),
                              ('tcn','hybrid'),('pointwise','hybrid')]:
            status(f'decoder_{decoder}_{inputs}_s{seed}',dict(state='queued',gpu=gpu,seed=seed))
    execute(gpus[0],'prepare',['prepare','--cache',str(cache),'--source',a.source])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu,seed,a,cache) for seed,gpu in enumerate(gpus)]
        for future in futures:
            future.result()
    subprocess.run([sys.executable,str(ROOT/'experiments/decoder_diagnostics.py'),
                    '--runs',str(ROOT/'logs/camels531'),'--out',str(OUT/'diagnostics')],
                    cwd=ROOT,check=True)
    (OUT/'complete.json').write_text(json.dumps(dict(completed=stamp())))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpus',default='2,4')
    p.add_argument('--source',required=True)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--detach',action='store_true')
    try:
        main(p.parse_args())
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'failed.json').write_text(json.dumps(dict(failed=stamp(),error=str(exc))))
        raise
