"""TEMPLATE — training the assembled model. NOT IMPLEMENTED.

Four stages, each runnable and checkable before the next begins.

STAGE 0 — unit-wise pretraining. DEBUGGING AND INITIALISATION.
    Each encoder alone on its own masked objective. Two purposes: clear that
    unit's U3 gate, and provide initialisation weights for stage 1. It is NOT
    where the final weights come from.
        done   unit B  `train_dem_sampler.py`
        done   unit D  `train_measurement_pfn.py`
        free   unit A  StefaLand checkpoint (ICDS, see docs/architecture.md)
        todo   unit C

STAGE 1 — per-site joint. All modalities in ONE sequence.
    Initialise every unit from stage 0. Cross-modal masking from the start.
    This is where attributes reach unit C, and it is what StefaLand already
    does per-site. Training jointly avoids the sequential hazard of freezing:
    a frozen A propagates its limitations into C and then into the connector,
    and once the connector changes what A *should* encode, C's learned reliance
    on it is stale.

STAGE 2 — cross-site. Add the connector.
    Freeze or LoRA the per-site encoders; train the connector on task-sampled
    batches. WHOLE-SITE masking is what forces neighbour use: with every
    observation at the query site hidden, other sites are the only information
    left. RANDOMISE THE CONTEXT SIZE here — measured, a fixed `n_ctx` makes the
    model calibrate off one neighbour and never learn to aggregate.

STAGE 3 — joint fine-tune. Everything, low LR, residual adapters.
    StefaLand ships adapters; use them rather than hard freezing so
    representations can still shift. Missing-modality dropout throughout, so
    any subset of inputs works at inference. The only stage needing a
    multi-GPU machine.

A BATCH IS A TASK, not a row: sample S sites, sample a mask pattern, mask,
reconstruct. Loss = sum of per-modality reconstruction in DATA space, weighted
so no modality dominates by sheer token count. Mask ratio on a schedule,
easy -> hard.

Splits by SITE, leave-region-out, throughout. Non-negotiable — every number in
this repository was produced that way, and the one time a split was defined by
COMID instead it silently duplicated 1,360 sites across both arms.
"""

from __future__ import annotations

import argparse


def main(cfg) -> None:
    raise NotImplementedError("see the module docstring for the staged plan")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, choices=[0, 1, 2, 3], required=True)
    main(ap.parse_args())
