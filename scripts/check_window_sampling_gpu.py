"""Real-cache equivalence and full-size online-training smoke on one GPU."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import camels531_decoders as d


def main(a):
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    cache=Path(a.cache)
    m=json.loads((cache/'manifest.json').read_text())
    data=dict(np.load(cache/'data.npz'))
    feats=np.load(cache/'train_features.npy',mmap_mode='r')
    enc=d.load_frozen_encoder(a.source,m,data,'cuda')
    b=np.arange(128)
    start=int(data['train_starts'][0])
    _,_,cached,_=d.make_inputs(data,feats,0,b,m['length'],m['patch'],'cuda')
    before=torch.cuda.get_rng_state().clone()
    _,_,online,_=d.online_inputs(enc,data,b,start,m,32,'cuda')
    # Strictly compare crop construction on this SAME device/runtime.
    # Historical cache values came from a different GPU/CUDA/torch backend.
    reference=[]
    for lo in range(0,len(b),32):
        batch=d.encoder_batch(data['forcing'][b[lo:lo+32]],data['attrs'][b[lo:lo+32]],
                              start,m['length'],m['patch'],'cuda')
        with torch.no_grad():
            reference.append(enc(batch,return_hidden=True)[:,:,-1])
    reference=d.lift_patch_features(torch.cat(reference),m['patch'])
    torch.testing.assert_close(reference,online,rtol=0,atol=0)
    same_device_error=float((reference-online).abs().max())
    torch.testing.assert_close(cached,online,rtol=1e-5,atol=1e-4)
    torch.testing.assert_close(before,torch.cuda.get_rng_state(),rtol=0,atol=0)
    error=float((cached-online).abs().max())
    del cached,online,reference,batch
    digest=d.weights_sha256(enc)
    rows=[]
    for kind in ('lstm','tcn','patch'):
        net=(d.PatchReconstructionDecoder(m['latent_dim'],m['patch']) if kind=='patch'
             else d.DailySequenceDecoder(5,26,kind,latent_dim=m['latent_dim'])).cuda().train()
        opt=torch.optim.Adadelta(net.parameters(),lr=1.)
        torch.cuda.reset_peak_memory_stats()
        elapsed=[]
        for step in range(4):
            torch.cuda.synchronize();begin=time.perf_counter()
            s=start+step+1
            x,att,z,_=d.online_inputs(enc,data,b,s,m,32,'cuda')
            target=(data['q_raw'][b,s+m['warmup']:s+m['length']]-data['q_mean'])/data['q_std']
            y=torch.tensor(target,device='cuda')
            pred=net(x,att,z)[:,m['warmup']:]
            loss=d.basin_normalized_mse(pred,y,torch.isfinite(y),
                                      torch.tensor(data['variance'][b],device='cuda'))
            assert torch.isfinite(loss)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            torch.cuda.synchronize()
            if step: elapsed.append(time.perf_counter()-begin)
        rows.append(dict(decoder=kind,seconds_per_online_step=float(np.mean(elapsed)),
                         peak_memory_bytes=torch.cuda.max_memory_allocated(),finite_loss=float(loss)))
        del opt,net,x,att,z,pred,loss,y
    assert d.weights_sha256(enc)==digest
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(),historical_cache_max_feature_error=error,
                         same_device_crop_max_feature_error=same_device_error,
                         encoder_unchanged=True,dropout_rng_unchanged_by_encoding=True,
                         batch=128,window=m['length'],encoder_chunk=32,smoke=rows),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True)
    p.add_argument('--source',required=True)
    main(p.parse_args())
