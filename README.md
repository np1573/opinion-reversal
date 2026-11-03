# Opinion Reversal on Complex Networks

Code and data for the paper:

**"Endogenous susceptibility and the response boundary of opinion reversal on complex networks"**

Ruining Wang, Canmin Zhang, Tianze Zhang, Xiaoyu Qian, Ya Zhou

*School of Systems Science, Beijing Normal University, Beijing, China*

> Status: prepared for submission to *Physica A: Statistical Mechanics and its Applications*.

## Overview

This repository contains the simulation code and result data for studying **opinion reversal** — the abrupt swing of collective opinion under counter-information — as a state-dependent response of complex systems.

The model implements opinion dynamics on complex networks (Barabási–Albert scale-free, Erdős–Rényi, and Watts–Strogatz), in which each agent's susceptibility to counter-information is a path-dependent state variable that grows with the speed and intensity of local consensus formation. Monte Carlo simulation maps the reversal probability onto a response diagram and reveals a decreasing response boundary `I_c(χ_pre)` separating the reversal and non-reversal regimes.

## Repository structure

```
.
├── code/
│   ├── model.py                  # core model & Monte Carlo simulation
│   ├── experiments.py            # main experiment driver (baseline / ablation /
│   │                             #   consensus / phase / info / robust / all)
│   ├── bench.py                  # benchmark
│   ├── smoke.py                  # smoke test
│   ├── verify_L1.py              # supplementary verifications
│   ├── verify_L1_window.py
│   ├── verify_peak_delay.py
│   ├── output/                   # main-experiment result data
│   └── supplementary/            # supplementary experiments & figure scripts
│       ├── model.py              # (identical to code/model.py)
│       ├── exp_*.py              # supplementary experiments (state swap,
│       │                         #   alternative susceptibility, mechanism
│       │                         #   decomposition, etc.)
│       ├── plot_*.py             # figure-generation scripts
│       ├── rebuild_*.py          # figure rebuild scripts
│       ├── run_*.py              # batch-run drivers (incl. N = 4000)
│       ├── analyze_*.py          # analysis scripts
│       ├── fss_*.py              # finite-size scaling
│       ├── verify_*.py           # verification scripts
│       └── output/               # supplementary result data
├── requirements.txt
└── LICENSE
```

## Requirements

Python 3.8+ with:

```
networkx
numpy
pandas
matplotlib
scipy
statsmodels
tqdm
Pillow
```

Install with:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# run the core experiment suite (or a single task)
python code/experiments.py all
python code/experiments.py baseline

# run supplementary scripts (from within the supplementary directory)
cd code/supplementary
python exp_state_swap.py
```

> Note: the model uses `multiprocessing` with the `fork` start method (set in `model.py`), which is supported on Linux and macOS.

## Data

All numeric results underlying the paper's figures are stored alongside the
scripts that produce them (`code/output/` and `code/supplementary/output/`).
Figure scripts read their inputs from the local `output/` directory. The
large per-run bootstrap archives (e.g. `fss200/` and `phase10boot/`) are
excluded; only the summarized results are included.

## Citation

If you use this code or data, please cite:

```
Ruining Wang, Canmin Zhang, Tianze Zhang, Xiaoyu Qian, Ya Zhou.
"Endogenous susceptibility and the response boundary of opinion reversal on complex networks."
```

## License

MIT License. See [LICENSE](LICENSE).
