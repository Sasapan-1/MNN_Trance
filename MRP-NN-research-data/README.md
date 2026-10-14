# MRP-NN Research Data

This repository provides trained neural-network checkpoints and saved evaluation data associated with research on implementing neural networks on a memory-based reconfigurable processor (MRP), prepared for submission to *Integration, the VLSI Journal*. It distinguishes MNIST classification results for the original neural networks (NNs), their truth-table networks, and saved RTL simulation outputs.

For the evaluation protocol, model configurations, and file descriptions, see [Evaluation Conditions](evaluation_conditions.md) (currently in Japanese).

Associated manuscript: A function synthesis and mapping framework for binarized neural networks on a memory-based reconfigurable processor

Corresponding author for the manuscript and repository maintainer / technical contact for data and code requests:  KENTA SASAGAWA　sasa.e.work.3735@gmail.com

## Evaluation results

All classification results are based on the full 10,000-image MNIST test set.

| Model | Original NN | Truth-table network | Matching MRP-end bits (NN vs. TT) | Classification from saved RTL outputs |
| --- | ---: | ---: | ---: | ---: |
| 6-layer MNN | 9,473/10,000 (94.73%) | 3,427/10,000 (34.27%) | 53,390/80,000 | Not evaluated |
| Corrected 6-layer BMNN | 9,240/10,000 (92.40%) | 9,240/10,000 (92.40%) | 80,000/80,000 | 9,240/10,000 (92.40%) |
| 11-layer Deep-BMNN | 9,229/10,000 (92.29%) | 9,229/10,000 (92.29%) | 80,000/80,000 | 9,229/10,000 (92.29%) |

The original MNN has continuous-valued intermediate activations. Its truth-table network uses binary inputs and outputs and is **not functionally equivalent to the original MNN**. Therefore, 34.27% is the classification accuracy of the *converted truth-table network*, not of the trained MNN or an equivalent RTL implementation.

For the RTL classification results, a separately saved 8-input, 10-output classification layer was applied in software to the eight MRP-end bits produced by the saved RTL simulations. **The complete 10-class classifier was not executed in RTL.**

## Files in this repository

Each `predictions.csv` file contains 10,000 test-sample indices, ground-truth labels, predicted labels, and correctness indicators. Each `mrp_end_bits.pt` is a PyTorch tensor of shape 10,000 × 8. The `.pth` checkpoints contain model states, training history, and optimizer states, but the model-definition source code is not included.

### 6-layer MNN

- Model checkpoint: [mnn6.pth](models/mnn6.pth)
- Original NN: [metrics.json](evaluation/mnn6/raw/metrics.json), [predictions.csv](evaluation/mnn6/raw/predictions.csv), [mrp_end_bits.pt](evaluation/mnn6/raw/mrp_end_bits.pt)
- Truth-table network: [metrics.json](evaluation/mnn6/truth/metrics.json), [predictions.csv](evaluation/mnn6/truth/predictions.csv), [mrp_end_bits.pt](evaluation/mnn6/truth/mrp_end_bits.pt)
- Bit comparison: [mrp_end_raw_vs_truth.json](evaluation/mnn6/comparisons/mrp_end_raw_vs_truth.json)

### Corrected 6-layer BMNN

- Model checkpoint: [bmnn6.pth](models/bmnn6.pth)
- Original NN: [metrics.json](evaluation/bmnn6/raw/metrics.json), [predictions.csv](evaluation/bmnn6/raw/predictions.csv), [mrp_end_bits.pt](evaluation/bmnn6/raw/mrp_end_bits.pt)
- Truth-table network: [metrics.json](evaluation/bmnn6/truth/metrics.json), [predictions.csv](evaluation/bmnn6/truth/predictions.csv), [mrp_end_bits.pt](evaluation/bmnn6/truth/mrp_end_bits.pt)
- Bit comparison: [mrp_end_raw_vs_truth.json](evaluation/bmnn6/comparisons/mrp_end_raw_vs_truth.json)
- Saved RTL results: [full10000.mem](rtl_results/bmnn6/full10000.mem), [evaluation.json](rtl_results/bmnn6/evaluation.json), [predictions.csv](rtl_results/bmnn6/predictions.csv)

### 11-layer Deep-BMNN

- Model checkpoint: [deep_bmnn11.pth](models/deep_bmnn11.pth)
- Original NN: [metrics.json](evaluation/deep_bmnn11/raw/metrics.json), [predictions.csv](evaluation/deep_bmnn11/raw/predictions.csv), [mrp_end_bits.pt](evaluation/deep_bmnn11/raw/mrp_end_bits.pt)
- Truth-table network: [metrics.json](evaluation/deep_bmnn11/truth/metrics.json), [predictions.csv](evaluation/deep_bmnn11/truth/predictions.csv), [mrp_end_bits.pt](evaluation/deep_bmnn11/truth/mrp_end_bits.pt)
- Bit comparison: [mrp_end_raw_vs_truth.json](evaluation/deep_bmnn11/comparisons/mrp_end_raw_vs_truth.json)
- Saved RTL results: [full10000.mem](rtl_results/deep_bmnn11/full10000.mem), [evaluation.json](rtl_results/deep_bmnn11/evaluation.json), [predictions.csv](rtl_results/deep_bmnn11/predictions.csv)

## Scope, reproducibility, and access to source code

This repository **does not include** the source code for training, inference, truth-table generation, MRP conversion and placement, or SystemVerilog verification. It also excludes RTL testbenches, simulation runners, and the original internal A/B working directories. Training, truth-table generation, and RTL simulation **cannot be reproduced using only the files in this repository**.

The underlying software and verification materials are **available upon reasonable request**. Please direct requests to the repository maintainer / technical contact listed above; access and sharing remain subject to applicable permissions and third-party rights. The manuscript's corresponding author can also coordinate requests with the current code maintainer.

The published RTL results are saved outputs from previous simulations, not records of a new RTL run performed as part of this release. In particular, the 11-layer RTL classification results were evaluated retrospectively from previously saved XSim outputs; the original manifest for the corresponding conversion inputs was not retained. Further details are provided in [Evaluation Conditions](evaluation_conditions.md).

## MNIST dataset

The original MNIST images are **not redistributed** in this repository. The dataset is available through its original distribution channels; see the [torchvision MNIST documentation](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html) for one way to obtain it.

## Rights and redistribution

No open redistribution license has yet been specified for the trained checkpoints or evaluation materials in this repository. The presence of files in a public GitHub repository does not itself grant permission to redistribute them outside the uses permitted by GitHub's Terms of Service and applicable law. For permissions beyond those uses, please contact the repository maintainer. This statement does not limit uses independently permitted by law.

