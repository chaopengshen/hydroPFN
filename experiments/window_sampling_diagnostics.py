"""Paired fixed-versus-daily sampling effects and hybrid/raw gap changes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from hydropfn.data import protocol as P

MODELS = [('lstm','raw'), ('tcn','raw'), ('lstm','hybrid'),
          ('tcn','hybrid'), ('patch','latent')]


def main(a):
    scores, rows, effects, details, gaps = {}, [], [], [], []
    reference_ids, reference_targets, reference_cache = None, None, None
    streams = {}
    for seed in (0,1):
        for decoder, inputs in MODELS:
            pair = []
            for sampling in ('fixed','daily'):
                tag = f'window_{decoder}_{inputs}_{sampling}_s{seed}'
                folder = Path(a.runs)/tag
                record = json.loads((folder/'run.json').read_text())
                cfg = record['config']
                ids, target = np.load(folder/'gage.npy'), np.load(folder/'targ.npy')
                if reference_ids is None:
                    reference_ids, reference_targets = ids, target
                    reference_cache = cfg['cache_manifest_sha256']
                np.testing.assert_array_equal(ids, reference_ids)
                np.testing.assert_allclose(target, reference_targets, rtol=0, atol=0, equal_nan=True)
                assert cfg['cache_manifest_sha256'] == reference_cache
                pred = np.load(folder/'pred.npy')
                assert np.isfinite(pred).all(), tag
                nse = np.asarray(P.nse_table(pred,target).nse)
                np.testing.assert_allclose(np.nanmedian(nse),record['median_nse'],rtol=0,atol=0)
                scores[(seed,decoder,inputs,sampling)] = nse
                assert cfg['feature_mode'] == 'online'
                if inputs != 'raw':
                    assert record['encoder_unchanged'], tag
                stream = record['paired_sample_stream_sha256']
                if seed in streams:
                    assert streams[seed] == stream, 'basin/random stream differs across treatments'
                streams[seed] = stream
                pair.append(record)
                rows.append(dict(tag=tag, seed=seed, decoder=decoder, inputs=inputs,
                                 sampling=sampling, median_nse=record['median_nse'],
                                 unique_starts=record['unique_train_starts'],
                                 training_seconds=record['training_seconds']))
                details.extend(dict(tag=tag,gage=str(g),nse=float(v)) for g,v in zip(ids,nse))
            f,d = pair
            for key in ('initial_weights_sha256','gpu_name','cache_manifest_sha256',
                        'epochs','batch','hidden','tcn_width','tcn_blocks','dropout','lr',
                        'feature_mode','online_encoder_initial_sha256'):
                assert f['config'][key] == d['config'][key], key
            assert f['config']['runtime']['slurm_job_id'] == d['config']['runtime']['slurm_job_id']
            assert f['optimizer_updates'] == d['optimizer_updates'] == 38800
            nf,nd = [scores[(seed,decoder,inputs,s)] for s in ('fixed','daily')]
            good = np.isfinite(nf) & np.isfinite(nd)
            effects.append(dict(seed=seed, decoder=decoder, inputs=inputs,
                median_nse_change=float(np.nanmedian(nd)-np.nanmedian(nf)),
                median_paired_basin_change=float(np.median(nd[good]-nf[good])),
                fraction_improved=float(np.mean(nd[good]>nf[good]))))
        for decoder in ('lstm','tcn'):
            raw_f,raw_d = [np.nanmedian(scores[(seed,decoder,'raw',s)]) for s in ('fixed','daily')]
            hyb_f,hyb_d = [np.nanmedian(scores[(seed,decoder,'hybrid',s)]) for s in ('fixed','daily')]
            gaps.append(dict(seed=seed,decoder=decoder,
                fixed_raw_minus_hybrid=float(raw_f-hyb_f),
                daily_raw_minus_hybrid=float(raw_d-hyb_d),
                gap_reduction=float((raw_f-hyb_f)-(raw_d-hyb_d)),
                raw_sampling_gain=float(raw_d-raw_f),hybrid_sampling_gain=float(hyb_d-hyb_f)))
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for name,data in [('scores',rows),('sampling_effects',effects),('gap_changes',gaps),('basin_scores',details)]:
        pd.DataFrame(data).to_csv(out/f'{name}.csv',index=False)
    (out/'audit.json').write_text(json.dumps(dict(
        n_basins=len(reference_ids),n_days=reference_targets.shape[1],
        target_gauge_cache_alignment='verified',initialization_and_basin_stream_pairs='verified',
        same_slurm_job_per_fixed_daily_pair=True,online_encoder_unchanged=True,
        interpretation='Positive gap_reduction means daily sampling narrows the hybrid deficit.',
        limits='Two seeds; fixed test diagnostics, no test-based checkpoint selection; frozen encoder only.'),indent=2))
    print(pd.DataFrame(effects).to_string(index=False),flush=True)
    print(pd.DataFrame(gaps).to_string(index=False),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',required=True)
    p.add_argument('--out',required=True)
    main(p.parse_args())
