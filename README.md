# Opinion Reversal on Complex Networks

Code and data for the paper:

**"Endogenous susceptibility and the response boundary of opinion reversal on complex networks"**

Ruining Wang, Canmin Zhang, Tianze Zhang, Xiaoyu Qian, Ya Zhou

*School of Systems Science, Beijing Normal University, Beijing, China*

> Status: under review at *Chaos, Solitons & Fractals*.

## Overview

This repository contains the simulation code and result data for studying **opinion reversal** — the abrupt swing of collective opinion under counter-information — as a state-dependent response of complex systems.

The model implements opinion dynamics on complex networks (Barabási–Albert scale-free, Erdős–Rényi, and Watts–Strogatz), in which each agent's susceptibility to counter-information is a path-dependent state variable that grows with the speed and intensity of local consensus formation. Monte Carlo simulation maps the reversal probability onto a response diagram and reveals a monotonically decreasing response boundary `I_c(χ_pre)`.

## Repository structure

```
.
├── code/
│   ├── model.py                 # core model & Monte Carlo simulation
│   ├── experiments.py           # experiment driver
│   ├── bench.py                 # benchmark
│   ├── smoke.py                 # smoke test
│   └── figures/                 # scripts to reproduce the paper figures
│       ├── plot_fig1_conceptual.py
│       ├── plot_fig3_ablation.py
│       ├── plot_fig7_ic_boundary.py
│       ├── plot_fig10_timeline.py
│       ├── rebuild_fss.py       # finite-size scaling
│       ├── fss_fixed_chi.py     # FSS with fixed susceptibility
│       ├── verify_ic_all.py     # response-boundary verification
│       └── run_N4000_remote.py  # N = 4000 large-scale run
├── data/                        # simulation results
│   ├── fss_N*.csv               # finite-size scaling data (N = 250–4000)
│   ├── fss_fixedchi_N*.csv      # FSS with fixed χ_pre
│   ├── phase5_*.npy / phase10_*.npy   # phase-diagram data
│   ├── robust_*.csv             # robustness checks
│   ├── speed_*.csv / *.json / *.npz   # consensus-formation speed data
│   ├── E1_*.npy / E2_fixedS.json      # case experiments
│   └── fig1_density.npz         # concept-figure data
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
```

Install with:

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# run the core model
python code/model.py

# run the experiment suite
python code/experiments.py
```

> Note: the model uses `multiprocessing` with the `fork` start method (set in `model.py`), which is supported on Linux and macOS.

## Data

All numeric results underlying the paper's figures are under `data/`. See the figure scripts in `code/figures/` for how each dataset is read and plotted.

## Citation

If you use this code or data, please cite:

```
Ruining Wang, Canmin Zhang, Tianze Zhang, Xiaoyu Qian, Ya Zhou.
"Endogenous susceptibility and the response boundary of opinion reversal on complex networks."
(under review at Chaos, Solitons & Fractals).
```

## License

MIT License. See [LICENSE](LICENSE).
