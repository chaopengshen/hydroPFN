"""Paired basin comparisons for the completed frozen-PFN decoder pilot."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hydropfn.data import protocol as P


def main(a):
    rows, details, comparisons = [], [], []
    reference_ids, reference_targets = None, None
    scores = {}
    for seed in a.seeds.split(','):
        for decoder, inputs in [('lstm','raw'),('tcn','raw'),('lstm','hybrid'),
                                ('tcn','hybrid'),('pointwise','hybrid')]:
            tag = f'decoder_{decoder}_{inputs}_s{seed}'
            folder = Path(a.runs)/tag
            record = json.loads((folder/'run.json').read_text())
            ids, target = np.load(folder/'gage.npy'), np.load(folder/'targ.npy')
            if reference_ids is None:
                reference_ids, reference_targets = ids, target
            np.testing.assert_array_equal(ids, reference_ids)
            np.testing.assert_allclose(target, reference_targets, rtol=0, atol=0, equal_nan=True)
            nse = np.asarray(P.nse_table(np.load(folder/'pred.npy'), target).nse)
            scores[(seed,decoder,inputs)] = nse
            rows.append(dict(tag=tag, seed=seed, decoder=decoder, inputs=inputs,
                             median_nse=float(np.nanmedian(nse)),
                             zero_latent_nse=record.get('zero_latent_median_nse'),
                             training_seconds=record['training_seconds'],
                             inference_seconds=record['inference_seconds'],
                             params=record['config']['n_parameters'],
                             gpu=record['config']['gpu_name']))
            details.extend(dict(gage=str(g),tag=tag,nse=float(v)) for g,v in zip(ids,nse))
        contrasts = [('lstm','hybrid','lstm','raw'),('tcn','hybrid','tcn','raw'),
                     ('tcn','hybrid','lstm','hybrid'),
                     ('lstm','hybrid','pointwise','hybrid'),('tcn','hybrid','pointwise','hybrid')]
        for d1,i1,d0,i0 in contrasts:
            x,y = scores[(seed,d1,i1)], scores[(seed,d0,i0)]
            good = np.isfinite(x) & np.isfinite(y)
            x,y = x[good],y[good]
            rng = np.random.default_rng(20260923)
            boot = []
            for _ in range(2000):
                take = rng.integers(len(x),size=len(x))
                boot.append(np.median(x[take])-np.median(y[take]))
            comparisons.append(dict(seed=seed, model=f'{d1}_{i1}', reference=f'{d0}_{i0}',
                      median_difference=float(np.median(x)-np.median(y)),
                      median_paired_delta=float(np.median(x-y)),fraction_better=float(np.mean(x>y)),
                      lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975))))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(out/'scores.csv',index=False)
    pd.DataFrame(details).to_csv(out/'basin_scores.csv',index=False)
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False)
    (out/'audit.json').write_text(json.dumps(dict(gauge_target_alignment='verified',
        n_basins=len(reference_ids),n_days=reference_targets.shape[1],
        interval_limit='paired basin bootstrap only; no spatial-dependence or seed uncertainty model',
        scope='K=0 frozen-PFN hindcast pilot, not daily DI or standard 730-day LSTM training',
        parity_certified=False),indent=2))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seeds',default='0,1')
    main(p.parse_args())
