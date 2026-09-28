"""365+365 raw LSTM recipe followed by frozen-baseline correction controls.

No validation selection, ensembles or joint encoder training. PFN retains its
native 512-day geometry and supplies features for the last 365 scored days.
Raw LSTM evaluation is continuous, identically reused by both corrections.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from camels531_lstm import RegionalLSTM
from camels531_decoders import (load_frozen_encoder, encoder_batch, sha,
                               weights_sha256, lift_patch_features)
from hydropfn.data import protocol as P
from hydropfn.models.residual_correction import ResidualCorrection
from hydropfn.train.objectives import basin_normalized_mse


def fit(a):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda' and not a.allow_cpu:
        raise RuntimeError('GPU required')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    cache = Path(a.cache)
    m = json.loads((cache/'manifest.json').read_text())
    for name, digest in m['files_sha256'].items():
        assert sha(cache/name) == digest, name
    d = dict(np.load(cache/'data.npz'))
    encoder = load_frozen_encoder(a.source, m, d, device)
    encoder_hash = weights_sha256(encoder)
    torch.manual_seed(a.seed)
    raw = RegionalLSTM(d['forcing'].shape[-1], d['attrs'].shape[-1], dropout=0.5).to(device)
    opt = torch.optim.Adadelta(raw.parameters(), lr=1.)
    rng = np.random.default_rng(a.seed)
    steps = a.steps or P.iters_per_epoch(len(d['gage']), m['train_stop']-m['train_start'], a.batch)
    config = dict(vars(a), steps_resolved=steps, warmup=365, rho=365, dropout=0.5,
                  optimizer='Adadelta lr=1 clip=1', encoder_sha256=encoder_hash,
                  cache_manifest_sha256=sha(cache/'manifest.json'),
                  evaluation='continuous raw LSTM; 512-day PFN windows with 365 scored days',
                  selection='fixed epochs; no validation or test selection', device=device)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    q = (d['q_raw']-float(d['q_mean']))/float(d['q_std'])
    target = d['q_raw'][:, m['score_start']:m['score_stop']]
    def tensor(x): return torch.as_tensor(x, device=device, dtype=torch.float32)
    def sample(rng):
        b = rng.choice(len(d['gage']), min(a.batch, len(d['gage'])), replace=False)
        s = int(rng.integers(m['train_start'], m['train_stop']-730+1))
        x = tensor(np.nan_to_num(d['forcing'][b, s:s+730]))
        attrs = tensor(d['attrs'][b])
        return b, s, x, attrs, tensor(q[b, s+365:s+730])
    def loss(p, y, b):
        return basin_normalized_mse(p, y, torch.isfinite(y), tensor(d['variance'][b]))
    def record(name, epoch, pred):
        physical = pred*float(d['q_std'])+float(d['q_mean'])
        nse = np.asarray(P.nse_table(physical, target).nse)
        np.save(out/f'{name}_e{epoch}_pred.npy', physical)
        np.save(out/f'{name}_e{epoch}_nse.npy', nse)
        row = dict(model=name, epoch=epoch, median_nse=float(np.nanmedian(nse)))
        with (out/'scores.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(row, flush=True)
    @torch.no_grad()
    def raw_eval():
        raw.eval()
        parts=[]
        for lo in range(0,len(d['gage']),16):
            x=tensor(np.nan_to_num(d['forcing'][lo:lo+16,m['score_start']-365:m['score_stop']]))
            parts.append(raw(x,tensor(d['attrs'][lo:lo+16]))[:,365:].cpu().numpy())
        return np.concatenate(parts)
    begin=time.time()
    for ep in range(1,a.raw_epochs+1):
        raw.train(); total=0.
        for _ in range(steps):
            b,s,x,attrs,y=sample(rng)
            value=loss(raw(x,attrs)[:,365:],y,b)
            opt.zero_grad(); value.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(),1.); opt.step()
            total+=value.item()
        print(dict(stage='raw',epoch=ep,loss=total/steps,seconds=time.time()-begin),flush=True)
        if ep in (50,100,a.raw_epochs):
            record('raw',ep,raw_eval())
            torch.save(raw.state_dict(),out/f'raw_e{ep}.pt')
    raw.eval().requires_grad_(False)
    raw_hash=weights_sha256(raw)
    baseline=raw_eval()
    np.save(out/'targ.npy',target); np.save(out/'gage.npy',d['gage'])
    # Same frozen raw model, same samples, same optimizer settings in both arms.
    @torch.no_grad()
    def features(b,s):
        parts=[]
        for lo in range(0,len(b),32):
            bb=b[lo:lo+32]
            batch=encoder_batch(d['forcing'][bb,s:s+512],d['attrs'][bb],0,512,m['patch'],device,time_offset=s)
            parts.append(encoder(batch,return_hidden=True)[:,:,-1])
        return lift_patch_features(torch.cat(parts),m['patch'])[:,-365:]
    # Cache only evaluation features, once. Every training crop is freshly encoded.
    windows=[]
    for lo in range(m['score_start'],m['score_stop'],365):
        hi=min(lo+365,m['score_stop']); start=hi-512
        windows.append((start,lo,hi,features(np.arange(len(d['gage'])),start).cpu()))
    @torch.no_grad()
    def correction_eval(head,use_pfn):
        head.eval(); result=np.empty_like(baseline)
        for start,lo,hi,z_cpu in windows:
            n=hi-lo; dest=slice(lo-m['score_start'],hi-m['score_start'])
            for b0 in range(0,len(d['gage']),32):
                b=slice(b0,b0+32)
                z=z_cpu[b,-n:].to(device) if use_pfn else None
                result[b,dest]=head(tensor(baseline[b,dest]),tensor(np.nan_to_num(d['forcing'][b,lo:hi])),
                                   tensor(d['attrs'][b]),z).cpu().numpy()
        return result
    audits={}
    for name,use_pfn in [('raw_correction',False),('pfn_correction',True)]:
        torch.manual_seed(a.seed+1000)
        rng=np.random.default_rng(a.seed+1000)
        head=ResidualCorrection(d['forcing'].shape[-1],d['attrs'].shape[-1],m['latent_dim'] if use_pfn else 0).to(device)
        initial=correction_eval(head,use_pfn)
        np.testing.assert_array_equal(initial,baseline)
        audits[name]=dict(initial_max_abs_difference=float(np.max(np.abs(initial-baseline))),
                          parameters=sum(p.numel() for p in head.parameters()))
        (out/'parity_audit.json').write_text(json.dumps(audits,indent=2))
        opt=torch.optim.Adadelta(head.parameters(),lr=1.)
        for ep in range(1,a.correction_epochs+1):
            head.train(); total=0.
            for _ in range(steps):
                b,s,x,attrs,y=sample(rng)
                with torch.no_grad(): base=raw(x,attrs)[:,365:]
                z=features(b,s+218) if use_pfn else None
                value=loss(head(base,x[:,365:],attrs,z),y,b)
                if not torch.isfinite(value): raise RuntimeError('nonfinite loss')
                opt.zero_grad(); value.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.); opt.step()
                total+=value.item()
            print(dict(stage=name,epoch=ep,loss=total/steps),flush=True)
            if ep in (10,25,50,a.correction_epochs):
                record(name,ep,correction_eval(head,use_pfn))
                torch.save(head.state_dict(),out/f'{name}_e{ep}.pt')
        assert weights_sha256(raw)==raw_hash
        assert weights_sha256(encoder)==encoder_hash
    (out/'complete.json').write_text(json.dumps(dict(raw_unchanged=True,encoder_unchanged=True,
                                                   seconds=time.time()-begin),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True); p.add_argument('--source',required=True)
    p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--raw-epochs',type=int,default=100)
    p.add_argument('--correction-epochs',type=int,default=50)
    p.add_argument('--steps',type=int,default=0); p.add_argument('--batch',type=int,default=128)
    p.add_argument('--allow-cpu',action='store_true')
    fit(p.parse_args())
