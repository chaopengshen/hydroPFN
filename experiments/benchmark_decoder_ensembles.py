"""Score fixed equal-weight ensembles; no weight fitting or best-epoch selection."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from hydropfn.data import protocol as P


def main(a):
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    seeds=[int(s) for s in a.seeds.split(',')]
    rows=[]
    ref_y=ref_g=None
    for epoch in [int(e) for e in a.epochs.split(',')]:
        predictions={}
        for model in ('lstm','tcn'):
            for seed in seeds:
                folder=Path(a.runs)/a.tag_template.format(model=model,seed=seed)
                r=json.loads((folder/'run.json').read_text())
                y,g=np.load(folder/'targ.npy'),np.load(folder/'gage.npy')
                if ref_y is None: ref_y,ref_g=y,g
                np.testing.assert_allclose(y,ref_y,rtol=0,atol=0,equal_nan=True)
                np.testing.assert_array_equal(g,ref_g)
                assert r['config']['sampling']=='daily'
                path=folder/('pred.npy' if epoch==r['config']['epochs'] else f'pred_epoch{epoch}.npy')
                pred=np.load(path)
                assert pred.shape==y.shape and np.isfinite(pred).all()
                predictions[(model,seed)]=pred
                rows.append(dict(epoch=epoch,name=f'{model}_s{seed}',kind='single',members=1,
                                 median_nse=float(np.nanmedian(P.nse_table(pred,y).nse))))
        groups={f'{model}_seed_mean':[(model,s) for s in seeds] for model in ('lstm','tcn')}
        groups['lstm_tcn_equal_mean']=list(predictions)
        for name,members in groups.items():
            pred=np.mean(np.stack([predictions[m] for m in members]),axis=0)
            nse=np.asarray(P.nse_table(pred,ref_y).nse)
            np.save(out/f'{name}_e{epoch}_pred.npy',pred)
            np.save(out/f'{name}_e{epoch}_nse.npy',nse)
            rows.append(dict(epoch=epoch,name=name,kind='equal_weight_ensemble',members=len(members),
                             median_nse=float(np.nanmedian(nse))))
    np.save(out/'gage.npy',ref_g)
    result=dict(scores=rows,n_basins=len(ref_g),n_days=ref_y.shape[1],
                target_alignment='verified',weights='equal; no test fitting',
                selection='All requested checkpoints reported; no best checkpoint selected',
                external_benchmark_beaten=None,
                limitation='531-basin Daymet K=0 hindcast, not the 671 table or multi-forcing benchmark')
    (out/'scores.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seeds',default='0,1,2')
    p.add_argument('--epochs',default='100,200,300')
    p.add_argument('--tag-template',default='benchmark_{model}_daily_s{seed}')
    main(p.parse_args())
