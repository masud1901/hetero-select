# HeteRo-Select

Official PyTorch implementation of **HeteRo-Select: Informativeness-Aware Client Selection and Gradient Compression for Communication-Efficient Federated Learning** (Masud, Jahin, & Hasan).

**Repository:** https://github.com/masud1901/hetero-select

HeteRo-Select drives client selection, compression ratio, local learning rate, and server aggregation from one normalized informativeness score; bandwidth is a hard ceiling only.

## Performance vs. State-of-the-Art (FedCG)

We evaluate against **FedCG** (Jiang et al., *Heterogeneity-Aware Federated Learning with Adaptive Client Selection and Gradient Compression*, **IEEE INFOCOM 2023** [1]). FedCG assigns compression ratios primarily from client bandwidth; HeteRo-Select allocates compression from statistical informativeness, with bandwidth as an upper bound only.

Under the matched 100-client simulation protocol on CIFAR-10 (mean ± std over seeds 42–44), HeteRo-Select reports:

| Metric | HeteRo-Select | FedCG (cited [1]) | Improvement |
| --- | --- | --- | --- |
| **Accuracy (Round 100)** | **72.56% ± 0.34%** | 70.00% | **+2.56 pts** |
| **Time to 70%** | **2,906s ± 41s** | 5,170s | **1.78× faster** |
| **Traffic to 70%** | **2,030 MB ± 19 MB** | 2,480 MB | **−18.2% less** |

**Stress test:** With bandwidth inverted against local loss (`--variant stress`), HeteRo-Select still reaches 70% on CIFAR-10 using **1,869 MB** simulated traffic (seed 42).

**CIFAR-100:** At round 100, HeteRo-Select uses about **59.6%** of FedCG's cited traffic and **45.2%** of cited simulated time (target 54% not reached in 100 rounds).

## Installation

**Requirements:** Python 3.10+, PyTorch 2.1+, torchvision 0.16+ (CUDA recommended).

```bash
pip install -e .
# or: pip install -r requirements.txt

# optional: conda env with pinned versions (see environment.yml)
# conda env create -f environment.yml && conda activate heteroselect
```

**Smoke test** (2 rounds, CPU, ~1 min):

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

Each run writes a JSON log to `results/<dataset>_psi<psi>_mu<mu>_seed<seed>_<variant>.json` containing per-round metrics and a `summary` block.

## Algorithm (code map and paper)

Each federated round:

1. **Score** all clients: loss (V), gradient diversity (D), fairness (F), staleness (St) → normalized `S_k` — `heteroselect/scoring.py`
2. **Select** M=10 clients via temperature-scaled softmax — `scoring.softmax_select`
3. **Compress** with cosine global budget `theta_t` and score-proportional per-client `theta_k` capped by bandwidth — `heteroselect/compression.py`
4. **Train** locally with FedProx and score-scaled learning rate — `heteroselect/client.py`
5. **Aggregate** with score-weighted server momentum — `heteroselect/server.py`

Orchestration: `heteroselect/trainer.py` · Entry point: `scripts/train.py`

Full mathematical specification, assumptions, and Algorithm 1 are in the paper (Sec. III). **Theoretical claims** (Theorems IV.1–IV.2) and **proofs** are in the paper (Sec. IV, Appendix A); this repository provides the empirical simulator only.

**Per-round complexity** (paper Sec. III.H): scoring costs O(K) forward passes on 8 mini-batches per client; training dominates at O(M·H·N) for M selected clients, H local steps, and N parameters, plus one Hutchinson Hessian-vector product per selected client on Q=3 Markov-sampled layers.

## Assumptions and simulation protocol

This code implements a **single-machine simulator** aligned with FedCG's **100-client simulation** (INFOCOM 2023, Sec. VI-C) [1]:

| Setting | Value |
| --- | --- |
| Clients K | 100 |
| Selected per round M | 10 |
| Local steps H | 50 |
| Rounds T | 100 |
| Local optimizer | FedProx, mu = 0.1 |
| Uplink bandwidth | B_k ~ Uniform[1, 5] Mb/s per client per round |
| Per-step compute time | T_k,cmp ~ Uniform[0.1, 0.5] s |
| Communication time (uplink) | theta_k · 32N / B_k |
| Round time | max over selected k of (t_com,k + H · T_k,cmp) |

Bandwidth is drawn uniformly each round (`variant=adaptive`). **`variant=stress`** assigns bandwidth inversely to normalized local loss (adversarial coupling test). **`variant=uniform`** uses a fixed per-client compression ratio `theta_k = theta_t`.

No real distributed network or multi-process FL is used; simulated time and traffic are analytic (see **Metrics**).

## Datasets and partitioning

| Dataset | Train / test | Classes | Model | Params | Target acc. | Source |
| --- | --- | --- | --- | ---: | ---: | --- |
| MNIST | 60k / 10k | 10 | Logistic regression | 7.85K | 90% | torchvision |
| CIFAR-10 | 50k / 10k | 10 | AlexNet | 2.78M | 70% | [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) |
| CIFAR-100 | 50k / 10k | 100 | ResNet9 | 6.62M | 54% | torchvision |
| TinyImageNet | 100k / 10k | 200 | ResNet-18 (64x64 stem) | 11.27M | 30% | [CS231N](http://cs231n.stanford.edu/tiny-imagenet-200.zip) |

**Splits:** Standard dataset train/test splits only; no validation hold-out; no examples excluded beyond the official test sets.

**Partitioning** (`heteroselect/data.py`, function `partition_data`):

- **MNIST / CIFAR-10:** psi-LDA-style — fraction psi of each client's ~500 samples from one dominant class, remainder spread over other classes (`psi=0.4` primary).
- **CIFAR-100 / TinyImageNet:** Skewed-label — each client lacks psi classes (40 or 80 missing classes respectively; integer psi in config).

**Preprocessing:** Random crop (pad 4) + horizontal flip + normalization for RGB sets; MNIST tensor + normalize only. TinyImageNet: unzip to `data/tiny-imagenet-200/` (not auto-downloaded).

**BatchNorm:** After aggregation, BN statistics are recalibrated on up to 20 batches from the first three selected clients for CIFAR-100 and TinyImageNet (`server.calibrate_bn`).

## Metrics

All test metrics use the **global model on the official test loader** after each round (`server.evaluate`).

| Field (JSON `summary`) | Definition |
| --- | --- |
| `peak_acc` | Maximum test accuracy over all rounds |
| `final_acc` | Test accuracy at round T |
| `mean_last10_acc` | Mean test accuracy over the last 10 rounds |
| `stability_drop` | `peak_acc` minus `final_acc` |
| `rounds_to_target` | First round where test acc >= dataset target (see `target_acc` in config); null if never |
| `time_to_target_s` | Cumulative **simulated** time at that round |
| `traffic_to_target_mb` | Cumulative **simulated uplink** traffic (MB) at that round |
| `total_sim_time_s` | Sum of per-round simulated wall times |
| `total_traffic_mb` | Sum over rounds of (sum over selected k of theta_k · model_mb) |
| `target_hit` | Whether `rounds_to_target` is set |

**Per-round JSON** also logs `sim_time_s`, `sim_traffic_mb`, `cum_time_s`, `cum_traffic_mb`, and **`wall_s`** (actual elapsed time on the machine running the simulator).

**Simulated traffic** does not include downlink or protocol overhead; it matches the paper's uplink-only accounting. **Simulated time** is not the same as `wall_s`; paper tables use simulated time/traffic unless noted.

**Targets** (`configs/default.yaml`): MNIST 90%, CIFAR-10 70%, CIFAR-100 54%, TinyImageNet 30%.

Aggregate multi-seed results:

```bash
python scripts/print_table.py results/experiment/
```

## Hyperparameters

Primary configuration is fixed **a priori** (no validation search). Defaults: `configs/default.yaml` and `heteroselect/config.py` (`DEFAULT_FL_CONFIG`).

| Symbol / key | Value | Role |
| --- | ---: | --- |
| `theta_total` | 0.20 | Mean compression budget (theta_avg) |
| `theta_floor` | 0.08 | Minimum theta_t |
| `warmup_rounds` | 1 | Round 1 uses full theta = 1 |
| `alpha_cos` | 0.4 | Cosine schedule on theta_t |
| `mu` | 0.1 | FedProx proximal coefficient |
| `local_lr` | 0.05 | Base eta_0 (decays over rounds) |
| `lr_scale_cap` | 0.15 | Cap on score-scaled LR |
| `lambda_V`, `lambda_D`, `lambda_F`, `lambda_St` | 1.0, 0.3, 0.2, 0.2 | Score weights |
| `gamma_St` | 0.5 | Staleness log scale |
| `tau_0` | 1.0 | Initial softmax temperature |
| `beta_s` | 0.5 | Server momentum |
| `newton_Q` | 3 | Markov-Newton layer budget |
| `newton_lambda` | 0.2 | Markov mixing rate |
| `grad_clip` | 2.0 | Global grad norm clip |
| `batch_size` | 32 | Local / eval mini-batch |
| `eval_batches` | 8 | Batches per client for scoring |

**Ablations** (one knob at a time, typically seed 42): `--variant uniform`, `--variant stress`, `--mu`, `--psi`, `--uniform-lr`, `--uniform-aggregation`, `--static-beta`, `--lambda-V` (and D/F/St), `--newton-Q`, `--tag` for output suffix.

## Seeds and experiment matrix

| Paper experiment | Seeds | Command |
| --- | --- | --- |
| Main table (MNIST, CIFAR-10/100, TinyImageNet) | 42, 43, 44 | `python scripts/train.py --grid main --results-dir results/experiment` |
| Ablations (compression, mu, psi, stress) | 42 | `python scripts/train.py --grid ablation --results-dir results/ablation` |
| Component ablations (noV, noD, ...) | 42 | Individual CLI flags + `--tag` (see `results/ablation_v2/`) |
| Non-IID sweeps | 42 | `--psi` overrides per run |

**Statistical reporting:** CIFAR-10/100 main comparisons use **mean ± std over three seeds** (42–44). Ablations and some auxiliary runs use **seed 42 only** unless stated otherwise.

## Baselines (FedCG et al. — cited, not re-run)

**FedAvg, OptRate, FlexCom, AdaSample, and FedCG** numbers and learning curves in the paper are **taken from the reported results** of Jiang et al. [1] (INFOCOM 2023 Table I, Figs. 1–3, and related simulation text), under the same 100-client simulation setting.

This repository **does not re-implement or re-run** those baselines. They are stored as:

- Scalar bars: `scripts/make_paper_figures.py` (`TABLE1`)
- Curve checkpoints: `figures-combined/` markdown tables (digitized from [1])
- INFOCOM-style 4-panel plots: `scripts/make_infocom_figures.py`

**Only HeteRo-Select** is executed via `scripts/train.py`. Comparisons attribute differences to the method, not to a mismatched simulator reimplementation.

**CIFAR-10 target note:** FedCG simulation text [1, Sec. VI-C] reports **5,170 s** to **70%** accuracy; Table I in [1] lists resource to **74%** with **2,480 MB** traffic. Our headline comparison uses the **70%** target and cited **5,170 s / 2,480 MB** pairing documented in the paper.

## Reproducing results

Exact commands for primary runs (seed 42; see `results/` for bundled logs):

| Experiment | Command | Peak Acc. | Target reached | Sim time (s) | Sim traffic (MB) |
| --- | --- | ---: | --- | ---: | ---: |
| CIFAR-10 (main) | `python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42` | 73.03% | Round 83 | 2,913 | 2,010 |
| CIFAR-100 (main) | `python scripts/train.py --dataset cifar100 --psi 40 --seed 42` | 49.44% | N/A @ 100 | 4,456 | 5,015 |
| MNIST | `python scripts/train.py --dataset mnist --psi 0.4 --seed 42` | 91.94% | Round 5 | 118.3 | 0.61 |
| TinyImageNet | `python scripts/train.py --dataset tinyimagenet --psi 80 --seed 42` | 33.53% | Round 79 | 4,622 | 7,061 |
| Ablation: uniform | `python scripts/train.py --dataset cifar10 --variant uniform --seed 42` | 72.22% | Round 86 | 3,009 | 2,124 |
| Ablation: stress | `python scripts/train.py --dataset cifar10 --variant stress --seed 42` | 71.80% | Round 84 | 3,254 | 1,869 |

For **3-seed** paper averages on CIFAR-10/100, repeat with `--seed 43` and `--seed 44`, then:

```bash
python scripts/print_table.py results/experiment/
```

**Batch reproduction:**

```bash
python scripts/train.py --grid main --results-dir results/experiment
python scripts/train.py --grid ablation --results-dir results/ablation
python scripts/train.py --grid all --results-dir results/ --zip
```

## Regenerating figures

From HeteRo-Select JSON logs (after training or using bundled `results/`):

```bash
# summary table from any results directory
python scripts/print_table.py results/experiment/

# basic plots from experiment + ablation JSON
python scripts/make_figures.py --exp results/experiment/ --abl results/ablation/ --out figs/

# ICDM-style paper figures (overlays cited baselines from [1])
python scripts/make_paper_figures.py --exp results/experiment/ --abl results/ablation/ --out figs/

# INFOCOM 4-panel comparison style
python scripts/make_infocom_figures.py --exp results/experiment/ --out figs/
```

Requires `matplotlib` (and `scipy` for `make_infocom_figures.py`).

Bundled paper figures (if present): `figs/figure-1-resource-overhead.pdf` through `figure-7-stress-test.pdf`.

## Computing environment

Experiments were run on **Linux** with a **single NVIDIA GPU** and **CUDA-enabled PyTorch 2.1+**.

| Component | Version (tested band) |
| --- | --- |
| Python | 3.10 – 3.11 |
| PyTorch | >= 2.1 |
| torchvision | >= 0.16 |
| numpy | >= 1.24 |

Pinned example: see [`environment.yml`](environment.yml). Record your exact GPU model and driver when reporting reproduction (e.g. NVIDIA RTX 3090, CUDA 12.x).

**Typical wall time per seed (`wall_s` in logs):** ~10 min (MNIST), ~10–15 min (CIFAR-10), ~25–35 min (CIFAR-100), ~3–5 h (TinyImageNet). Simulated time/traffic in paper tables use the analytic model above, not `wall_s`.

Energy was not measured.

## Project layout

```text
heteroselect/       # core package (config, data, models, scoring, compression, trainer, ...)
scripts/
  train.py          # training driver
  print_table.py    # aggregate JSON summaries
  make_figures.py
  make_paper_figures.py
  make_infocom_figures.py
configs/default.yaml
tests/test_smoke.py
results/            # bundled per-run JSON logs (optional to regenerate)
figures-combined/   # digitized baseline curves from [1]
figs/               # generated paper figures (optional)
```

## References

[1] Z. Jiang, Y. Xu, H. Xu, Z. Wang, and C. Qian, "Federated learning with client selection and gradient compression in heterogeneous edge systems," in *IEEE INFOCOM*, 2023. (Extended journal version: *IEEE Trans. Mobile Comput.*, 2024.)

## Citation

```bibtex
@article{masud2026heteroselect,
  title   = {HeteRo-Select: Informativeness-Aware Client Selection and Gradient
             Compression for Communication-Efficient Federated Learning},
  author  = {Masud, M. A. and Jahin, Md Abrar and Hasan, M.},
  journal = {Manuscript under review},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
