"""Generate eval_external.py: the ORIGINAL harness, able to load our ckpts.

Why a patched COPY and not a new script: every DEM regression this week came
from re-implementing a working pipeline instead of extending it. This script
performs ONE surgical replacement on test_diffusion_sampler.py -- the
checkpoint-loading block -- and leaves the eval protocol (tile split, seed,
mask draws, scoring) byte-identical. Run with the same --seed and the eval
draw is the SAME 60 tiles and masks that produced the champion's 0.824, so
the comparison is finally same-protocol.

The shim: our multi-scale checkpoints store no "width"/"T" (inferred from
weights), carry an smlp (scale conditioning), and expect a scale vector. A
tiny adapter injects the constant fine-scale conditioning (10 m / 1.28 km)
so the harness's net(x, t) calls need no changes.
"""
import re
import sys

SRC = "/nfs/data/cxs1024/dem_foundation/src/tests/test_diffusion_sampler.py"
DST = "/nfs/data/cxs1024/dem_foundation/src/tests/eval_external.py"

s = open(SRC, encoding="utf-8").read()

old = """    if ckpt:
        st = torch.load(ckpt, map_location=DEVICE)
        net = DenoiseUNet(st["width"], in_ch=3).to(DEVICE)
        net.load_state_dict(st["ema"]); net.eval()
        param = st.get("param", "eps")
        residual = st.get("residual", False)
        dif = Diffusion(st["T"], DEVICE, param=param)
        print(f"loaded {ckpt}  (param={param}, residual={residual})")"""

new = """    if ckpt:
        st = torch.load(ckpt, map_location=DEVICE)
        sd = st["ema"] if "ema" in st else st["net"]
        w_ = sd["inp.weight"].shape[0]
        in_ch = sd["inp.weight"].shape[1]
        has_scale = any(key.startswith("smlp.") for key in sd)
        if has_scale:
            # hydroPFN multi-scale checkpoint: build its own net class and
            # freeze the scale conditioning at this corpus's true values
            # (10 m/px, 1.28 km footprint). net(x, t) then works unchanged.
            sys.path.insert(0, "/nfs/data/cxs1024/hydroPFN/src")
            from hydropfn.models.diffusion import DenoiseUNet as MSUNet
            inner = MSUNet(w=w_, in_ch=in_ch, scale_cond=True).to(DEVICE)
            inner.load_state_dict(sd); inner.eval()
            import math as _math

            class _ScaleAdapter(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                    self.register_buffer("sc", torch.tensor(
                        [[_math.log10(10.0), _math.log10(1.28)]],
                        dtype=torch.float32))

                def forward(self, x, t):
                    return self.m(x, t, self.sc.expand(x.shape[0], -1))

            net = _ScaleAdapter(inner).to(DEVICE); net.eval()
        else:
            net = DenoiseUNet(st.get("width", w_), in_ch=in_ch).to(DEVICE)
            net.load_state_dict(sd); net.eval()
        param = st.get("param", "eps")
        residual = st.get("residual", False)
        dif = Diffusion(st.get("T", 1000), DEVICE, param=param)
        print(f"loaded {ckpt}  (param={param}, residual={residual}, "
              f"w={w_}, scale_cond={has_scale}, ema={'ema' in st})")"""

assert old in s, "loading block did not match -- original changed?"
s = s.replace(old, new, 1)
if "\nimport sys" not in s:
    s = s.replace("\nimport time", "\nimport sys\nimport time", 1)
open(DST, "w", encoding="utf-8").write(s)
print(f"wrote {DST}")
