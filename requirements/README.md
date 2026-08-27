# Candidate runtime environment

The initial compatibility target for the MMDetection 3.3 spike is:

- Python 3.9
- PyTorch 2.0.1 with CUDA 11.8
- torchvision 0.15.2 with CUDA 11.8
- MMCV 2.1.0
- MMEngine 0.10.5
- repository-local MMDetection 3.3.0
- NumPy 1.x (`numpy>=1.23,<2`) for the PyTorch 2.0.1 compiled ABI

This is a candidate, not a claim that the environment has passed the GPU
acceptance gate. Keep it isolated from other projects. Do not update the
current environment in place.

Install PyTorch and torchvision from the official CUDA 11.8 wheel index, then
install the matching prebuilt MMCV wheel with OpenMIM. Avoid an implicit MMCV
source build, especially on native Windows. Install the remaining runtime
packages only after the CUDA stack is selected.

`tqdm` is a direct runtime dependency used only for the compact terminal
progress line. It does not change the resolved scientific configuration,
training schedule, or artifact formats.

```powershell
python -m pip install torch==2.0.1 torchvision==0.15.2 `
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install openmim
mim install mmcv==2.1.0
python -m pip install -r requirements/runtime.txt
python tools/check_environment.py
```

The final requirement pins must be retained only if `check_environment.py`
passes on the target GPU, including CUDA NMS and RoIAlign backward. Linux or
WSL2 is the preferred GPU runtime. `--allow-cpu` is only for local structural
checks and does not satisfy the GPU acceptance gate.
