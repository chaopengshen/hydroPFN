# Torch env for suntzu.  Two traps, both hit in practice:
#  - pytorch_gpu lives under /data/cxs1024/tools/anaconda3, NOT the system
#    conda base, so "conda activate pytorch_gpu" fails and silently leaves you
#    on the base CPU torch (1.12, cuda unavailable) -- a run then reports "cpu"
#    in its banner and takes 3 min/epoch instead of seconds.
#  - libcusparse needs the pip nvjitlink shim ahead of the system one, else
#    import torch dies on __nvJitLinkAddData_12_1.  The path must be literal;
#    an unexpanded python3.* glob fails the same way.
export TORCH_ENV=/data/cxs1024/tools/anaconda3/envs/pytorch_gpu
export LD_LIBRARY_PATH=$TORCH_ENV/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # without this, CUDA_VISIBLE_DEVICES=2 lands on a 2080 Ti
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}   # 2 = RTX 3090 Ti, 24 GB
export PY=$TORCH_ENV/bin/python
