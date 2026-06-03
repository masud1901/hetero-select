# HeteRo-Select

Official PyTorch implementation of **HeteRo-Select: Informativeness as the Participation Driver in Heterogeneous Federated Learning**.

---

HeteRo-Select drives client selection, compression ratio, local learning rate, and server aggregation from one normalized informativeness score; bandwidth is a hard ceiling only.

## Performance vs. State-of-the-Art (FedCG)

We directly evaluate against **FedCG** (Jiang et al., *Heterogeneity-Aware Federated Learning with Adaptive Client Selection and Gradient Compression*, **IEEE INFOCOM 2023**). While FedCG assigns compression ratios primarily based on client bandwidth, HeteRo-Select allocates bandwidth based on *statistical informativeness*.

Under an identical 100-client simulation protocol on CIFAR-10 (mean ± std over 3 seeds), HeteRo-Select achieves three simultaneous wins over FedCG:

| Metric | HeteRo-Select | FedCG (INFOCOM '23) | Improvement |
|---|---|---|---|
| **Accuracy (Round 100)** | **72.56% ± 0.34%** | 70.00% | **+2.56 pts** |
| **Time to 70%** | **2,906s ± 41s** | 5,170s | **1.78× faster** |
| **Traffic to 70%** | **2,030 MB ± 19 MB** | 2,480 MB | **−18.2% less** |

**Robustness Under Adversarial Bandwidth:** Under a stress test where the most informative clients are placed on the slowest links (1 Mbps), a bandwidth-driven design like FedCG systematically degrades the most valuable gradients. HeteRo-Select remains robust, reaching the 70% target using only **1,869 MB** of traffic—outperforming FedCG's normal bandwidth results.

**CIFAR-100 Efficiency:** HeteRo-Select completes 100 rounds using only **59.6%** of FedCG's traffic and **45.2%** of FedCG's time.

## Requirements

- Python 3.10+
- PyTorch 2.x (CUDA recommended)

```bash
pip install -e .
# or: pip install -r requirements.txt
```

## Quick start

```bash
# smoke test (5 rounds)
python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42 --rounds 5

# primary CIFAR-10 run (100 rounds)
python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42
```

Logs are written to `results/<dataset>_psi<psi>_mu<mu>_seed<seed>_<variant>.json`.

## Datasets

| Dataset | Train / test | Classes | Model | Params | Target | Source |
|---------|--------------|---------|-------|-------:|-------:|--------|
| MNIST        | 60k / 10k  | 10  | Logistic regression | 7.85K | 90% | torchvision |
| CIFAR-10     | 50k / 10k  | 10  | AlexNet             | 2.78M | 70% | [torchvision](https://www.cs.toronto.edu/~kriz/cifar.html) |
| CIFAR-100    | 50k / 10k  | 100 | ResNet9             | 6.62M | 54% | torchvision |
| TinyImageNet | 100k / 10k | 200 | ResNet-18 (64x64 stem) | 11.27M | 30% | [Stanford CS231N](http://cs231n.stanford.edu/tiny-imagenet-200.zip) |

CIFAR-10/100 and MNIST are downloaded automatically. TinyImageNet must be unzipped into `data/tiny-imagenet-200/`. Preprocessing: random crop (pad 4), horizontal flip, dataset normalization. No extra cleaning or exclusions.

## Computing environment

- Linux, single NVIDIA GPU, CUDA-enabled PyTorch 2.1+
- Python 3.10–3.11, `torchvision` 0.16+
- Typical wall time: ~10 min/seed (MNIST), ~10–15 min/seed (CIFAR-10), ~25–35 min/seed (CIFAR-100), ~3–5 h/seed (TinyImageNet)

## Hyperparameters

Primary run uses fixed values in `configs/default.yaml` (no validation search): theta_avg=0.20, mu=0.1, M=10, H=50, T=100. Per-dataset partition: psi=0.4 (MNIST, CIFAR-10), 40 missing classes (CIFAR-100), 80 missing classes (TinyImageNet). Main table uses seeds 42–44; cross-scale and ablations use seed 42. Ablations vary one knob at a time (`--variant uniform|stress`, `--mu`, `--psi`).

## Reproducibility: Results and Commands

To comply with reproducibility guidelines, the table below provides the exact commands used to generate the reported experimental results, directly accompanied by their respective outcomes (using the primary configuration on `seed 42`). 

| Experiment | Precise Command | Peak Acc. | Target Reached | Sim Time (s) | Sim Traffic (MB) |
|---|---|---|---|---|---|
| **CIFAR-10 (Main)** | `python scripts/train.py --dataset cifar10 --psi 0.4 --seed 42` | 73.03% | Round 83 | 2,913 | 2,010 |
| **CIFAR-100 (Main)** | `python scripts/train.py --dataset cifar100 --psi 40 --seed 42` | 49.44% | *N/A (Rnd 100)* | 4,456* | 5,015* |
| **MNIST** | `python scripts/train.py --dataset mnist --psi 0.4 --seed 42` | 91.94% | Round 5 | 118.3 | 0.61 |
| **TinyImageNet** | `python scripts/train.py --dataset tinyimagenet --psi 80 --seed 42` | 33.53% | Round 79 | 4,622 | 7,061 |
| **Ablation: Uniform** | `python scripts/train.py --dataset cifar10 --variant uniform --seed 42` | 72.22% | Round 86 | 3,009 | 2,124 |
| **Ablation: Stress Test** | `python scripts/train.py --dataset cifar10 --variant stress --seed 42` | 71.80% | Round 84 | 3,254 | 1,869 |

*\*CIFAR-100 target (54%) not reached in 100 rounds; cumulative time/traffic reported at round 100. See paper Section VII-E.*  
*(Note: For the 3-seed averages reported in Table I of the paper, repeat the CIFAR-10/100 commands with `--seed 43` and `--seed 44` and average the results.)*

### Batch Reproduction

You can batch all the runs automatically instead of executing them individually:

```bash
python scripts/train.py --grid main      # 3-seed MNIST / CIFAR-10 / CIFAR-100 / TinyImageNet
python scripts/train.py --grid ablation  # uniform, mu, psi, stress
python scripts/train.py --grid all --zip
```

Once completed, you can automatically regenerate all paper tables and figures from the output JSON logs:

```bash
python scripts/print_table.py results/
python scripts/make_figures.py --exp results/ --abl results/ --out figs/
```

## Project layout

```
heteroselect/   # core package (config, data, scoring, compression, trainer, ...)
scripts/        # train.py, print_table.py, make_figures.py
configs/        # default.yaml
tests/          # smoke test
```

Key defaults live in `heteroselect.config.DEFAULT_FL_CONFIG` (or `configs/default.yaml`).


