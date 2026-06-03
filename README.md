# HeteRo-Select

Official PyTorch implementation of **HeteRo-Select: Informativeness as the Participation Driver in Heterogeneous Federated Learning** (Masud, Jahin, & Hasan).

**Repository:** https://github.com/masud1901/hetero-select

HeteRo-Select drives client selection, compression ratio, and server aggregation from one normalized informativeness score; bandwidth is a hard ceiling only. Local learning rate can be score-adaptive (primary runs) or uniform (ablation; see paper Sec. VI-B).

## Performance vs. State-of-the-Art (FedCG)

We evaluate against **FedCG** (Jiang et al., IEEE INFOCOM 2023 / IEEE TMC 2024 [1]). FedCG assigns compression ratios primarily from client bandwidth; HeteRo-Select allocates compression from statistical informativeness, with bandwidth as an upper bound only.

Under the matched 100-client simulation protocol on CIFAR-10 (mean ± std over seeds 42–44):

| Metric | HeteRo-Select | FedCG (cited [1]) | Improvement |
| --- | --- | --- | --- |
| **Accuracy (Round 100)** | **72.56% ± 0.34%** | 70.00% | **+2.56 pts** |
| **Time to 70%** | **2,906s ± 41s** | 5,170s | **1.78× faster** |
| **Traffic to 70%** | **2,030 MB ± 19 MB** | 2,480 MB | **−18.2% less** |

**Stress test:** `--variant stress` still reaches 70% on CIFAR-10 with **1,869 MB** simulated traffic (seed 42).

**CIFAR-100:** At round 100, HeteRo-Select uses about **59.6%** of FedCG's cited traffic and **45.2%** of cited simulated time (54% target not reached in 100 rounds).

## Installation

**Requirements:** Python 3.10+, PyTorch 2.1+, torchvision 0.16+ (CUDA recommended).

```bash
pip install -e .
# or: pip install -r requirements.txt

# optional: conda env (see environment.yml)
# conda env create -f environment.yml && conda activate heteroselect
```

**Smoke test** (2 rounds, CPU):

```bash
pytest tests/test_smoke.py
```

## Quick start

```bash
# short sanity run (5 rounds)
python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42 --rounds 5

# primary CIFAR-10 run (100 rounds)
python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42
```

Logs: `results/<dataset>_psi<psi>_mu<mu>_seed<seed>_<variant>.json`

## Algorithm (code map)

Each federated round (`heteroselect/trainer.py`):

1. **Score** — loss V, diversity D, fairness F, staleness St → `S_k` (`heteroselect/scoring.py`)
2. **Select** — M=10 via softmax (`scoring.softmax_select`)
3. **Compress** — cosine `theta_t`, score-proportional `theta_k` (`heteroselect/compression.py`)
4. **Train** — FedProx + optional score-scaled LR (`heteroselect/client.py`)
5. **Aggregate** — score-weighted server momentum (`heteroselect/server.py`)

**Ablations** are selected via `--variant` and implemented in `heteroselect/variants.py` (see table below).

Theory and proofs: paper Sec. IV / Appendix A.

## Assumptions and simulation protocol

Single-machine simulator aligned with FedCG's 100-client simulation [1, Sec. VI-C]:

| Setting | Value |
| --- | --- |
| Clients K | 100 |
| Selected M | 10 |
| Local steps H | 50 |
| Rounds T | 100 |
| FedProx mu | 0.1 |
| Bandwidth | B_k ~ Uniform[1, 5] Mb/s per client per round |
| Compute | T_k,cmp ~ Uniform[0.1, 0.5] s per local step |

Variants: `adaptive` (default), `uniform` (fixed theta_k), `stress` (bandwidth vs loss inverted).

## Datasets and partitioning

| Dataset | Train / test | Model | Target | Partition |
| --- | --- | --- | ---: | --- |
| MNIST | 60k / 10k | Logistic reg | 90% | psi-LDA, psi=0.4 |
| CIFAR-10 | 50k / 10k | AlexNet | 70% | psi-LDA, psi=0.4 |
| CIFAR-100 | 50k / 10k | ResNet9 | 54% | 40 missing classes |
| TinyImageNet | 100k / 10k | ResNet-18 | 30% | 80 missing classes |

TinyImageNet: unzip to `data/tiny-imagenet-200/`. See `heteroselect/data.py`.

## Metrics

Key JSON `summary` fields: `peak_acc`, `final_acc`, `rounds_to_target`, `time_to_target_s`, `traffic_to_target_mb`, `total_sim_time_s`, `total_traffic_mb`, `wall_s` (actual machine time).

Paper tables use **simulated** uplink time/traffic, not `wall_s`.

```bash
python scripts/print_table.py results/experiment/
```

## Hyperparameters

Defaults: `configs/default.yaml`, `heteroselect/config.py`.

Primary: `theta_total=0.20`, `warmup_rounds=1`, `alpha_cos=0.4`, `lambda_D/F/St=0.3/0.2/0.2`, `newton_Q=3`, `beta_s=0.5`, `mu=0.1`.

## Experiment variants (`--variant`)

Implemented in `heteroselect/variants.py`:

| Variant | Effect |
| --- | --- |
| `adaptive` | Full HeteRo-Select (primary) |
| `uniform` | Fixed compression theta_k = theta_t |
| `stress` | Bandwidth inversely proportional to local loss |
| `no_V` | Zero loss component V'_k |
| `no_D` | lambda_D = 0 |
| `no_FS` | lambda_F = lambda_St = 0 (fairness + staleness off) |
| `no_newton` | Pure magnitude top-k (newton_Q = 0) |
| `static_beta` | Fixed error-feedback beta = 0.90 |
| `uniform_lr` | Uniform local LR (ablation; faster in our runs) |
| `uniform_agg` | FedAvg-style aggregation weights |

CLI equivalents: `--uniform-lr`, `--uniform-aggregation`, `--static-beta 0.9`, `--newton-Q 0`, `--lambda-D 0`, etc.

## Seeds and experiment grids

| Grid | Command | Contents |
| --- | --- | --- |
| Main (seeds 42–44) | `python scripts/train.py --grid main --results-dir results/experiment` | MNIST, CIFAR-10/100, TinyImageNet |
| Protocol ablations | `python scripts/train.py --grid ablation --results-dir results/ablation` | uniform, mu, psi, stress |
| Component ablations (Table II) | `python scripts/train.py --grid component --results-dir results/ablation_v2` | no_V, no_D, no_FS, no_newton, static_beta, uniform_lr, uniform_agg |
| All | `python scripts/train.py --grid all --results-dir results/ --zip` | Union of the above |

## Baselines (cited, not re-run)

FedAvg, OptRate, FlexCom, AdaSample, and FedCG scalars/curves are **from Jiang et al. [1]** (digitized in `figures-combined/`). Only HeteRo-Select is executed in this repo.

## Reproducing paper results (seed 42)

### Main runs

| Experiment | Command | Peak | R→70% | Sim time | Sim traffic |
| --- | --- | ---: | --- | ---: | ---: |
| CIFAR-10 | `python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42` | 73.03% | 83 | 2,913 | 2,010 |
| CIFAR-100 | `python scripts/train.py --dataset cifar100 --psi 40 --seed 42` | 49.44% | — | 4,456 | 5,015 |
| MNIST | `python scripts/train.py --dataset mnist --psi 0.4 --seed 42` | 91.94% | 5 | 118.3 | 0.61 |
| TinyImageNet | `python scripts/train.py --dataset tinyimagenet --psi 80 --seed 42` | 33.53% | 79 | 4,622 | 7,061 |

### Table II component ablations (CIFAR-10, seed 42)

| Variant | Command suffix | Peak | Final | R→70% | Time | Traffic |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| w/o V'_k | `--variant no_V` | 72.71% | 72.71% | 85 | 2,972 | 2,050 |
| w/o D_k | `--variant no_D` | 72.25% | 71.10% | 85 | 2,868 | 2,047 |
| w/o F', St' | `--variant no_FS` | 72.08% | 71.75% | 82 | 2,850 | 1,998 |
| Pure magnitude top-k | `--variant no_newton` | 73.24% | 73.24% | 79 | 2,725 | 1,955 |
| Static beta=0.90 | `--variant static_beta` | 72.13% | 71.97% | 83 | 2,865 | 2,005 |
| Uniform LR | `--variant uniform_lr` | 74.08% | 74.08% | 70 | 2,458 | 1,802 |
| Uniform aggregation | `--variant uniform_agg` | 72.28% | 72.28% | 83 | 2,901 | 2,005 |

Bundled logs: `results/ablation_v2/`. Paper Sec. VI-B notes uniform LR can outperform score-adaptive LR in this ablation; primary Table I runs still use score-adaptive LR.

### Protocol ablations (Table II, upper block)

| Variant | Peak | R→70% | Time | Traffic |
| --- | ---: | ---: | ---: | ---: |
| Uniform compression | 72.22% | 86 | 3,009 | 2,124 |
| Adaptive (primary) | 73.03% | 83 | 2,913 | 2,010 |
| mu=0 | 64.36% | — | — | — |
| mu=0.01 | 71.92% | 81 | 2,864 | 1,967 |
| mu=0.5 | 54.57% | — | — | — |
| psi=0.2 | 73.88% | 76 | 2,746 | 1,921 |
| psi=0.6 | 68.18% | — | — | — |
| stress | 71.80% | 84 | 3,254 | 1,869 |

Bundled logs: `results/ablation/`.

## Figures

Bundled paper figures: `figs/figure-1-resource-overhead.pdf` … `figure-7-stress-test.pdf`.

Regenerate result tables from JSON:

```bash
python scripts/print_table.py results/experiment/
python scripts/print_table.py results/ablation_v2/
```

## Project layout

```text
heteroselect/
  config.py, data.py, models.py, scoring.py, compression.py
  client.py, server.py, trainer.py, simulator.py, variants.py
scripts/
  train.py          # all experiments and ablation grids
  print_table.py
configs/default.yaml
tests/test_smoke.py
results/            # bundled JSON logs
figures-combined/   # cited FedCG baseline curves
figs/               # bundled paper figures (gitignored; regenerate locally)
```

## References

[1] Z. Jiang et al., "Federated learning with client selection and gradient compression in heterogeneous edge systems," *IEEE INFOCOM*, 2023; *IEEE Trans. Mobile Comput.*, 2024.

## Citation

```bibtex
@inproceedings{masud2026heteroselect,
  title   = {HeteRo-Select: Informativeness as the Participation Driver in
             Heterogeneous Federated Learning},
  author  = {Masud, M. A. and Jahin, Md Abrar and Hasan, M.},
  booktitle = {IEEE International Conference on Data Mining (ICDM)},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
