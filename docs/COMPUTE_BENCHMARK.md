# Compute benchmark

Phase 2 preparation was run on the local Windows Python 3.13 environment
using CPU-only MNE preprocessing. The downloaded S22 MEG block is about
1.108 GB on disk; each extracted 50 Hz sentence window is approximately
260–300 time steps by 306 channels.

The one-trial CTC smoke test completed with finite logits and loss. The
six-trial, 40-epoch CPU overfit attempt took about 36.8 seconds and reduced
loss from 20.88 to 2.93, but predictions collapsed to spaces. A two-trial
150-epoch investigation took about 91.6 seconds and still did not reproduce
the sentences. These timings indicate that the compact baseline is
computationally practical for Colab-scale experiments, but the current
alignment/preprocessing/model combination is not yet validated for learning.
