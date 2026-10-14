# Evaluation Conditions and File Mapping

## Dataset and Evaluation Metrics

- The full 10,000-image standard MNIST test split is used, without shuffling during evaluation. Images are converted using `ToTensor` and normalized with a mean of 0.1307 and a standard deviation of 0.3081. The input images themselves are not binarized by thresholding.
- Classification accuracy is the number of correct predictions out of 10,000 samples, with the predicted class determined by the maximum of the 10 class outputs. The `index` field in `predictions.csv` is zero-based and follows the evaluation order.
- The output of the MRP-implemented portion of the network consists of 8 bits per sample. A total of 80,000 bits are compared between the original NN and the truth-table network. For the original NN, each terminal output value is converted to 1 if it is `> 0`, and to 0 otherwise, for this comparison.
- The RTL `full10000.mem` file contains one 8-bit terminal output per line, for each test sample. RTL classification accuracy is obtained by applying a separately saved 8-input, 10-output classification layer in software. It does **not** represent a result in which the entire 10-class classifier was executed in RTL.

## Models and Training

| Published model | Key architectural characteristics | Epoch stored in checkpoint | SHA-256 |
| --- | --- | ---: | --- |
| [models/mnn6.pth](models/mnn6.pth) | An external input layer converts 784 inputs into 48 values, followed by a 6-layer MRP portion. An external output layer converts the 8 terminal values into 10 classes. Intermediate values are continuous. | 700 | `3A1450819123F24AABAE79C952DB9ADDFED11140016C2242479CCA0EAEAB6E07` |
| [models/bmnn6.pth](models/bmnn6.pth) | The corrected model uses the same 48-value input, 6-layer MRP portion, and 8-value terminal output. MRP cell outputs are thresholded to 0 or 1. | 700 | `DBC5A246C62034314F232BA933256AA81973B86381F016DAD43B579E7AACFC5E` |
| [models/deep_bmnn11.pth](models/deep_bmnn11.pth) | An external input layer converts 784 inputs into 44 values, followed by an 11-layer MRP portion with a folded dataflow structure and an 8-value terminal output feeding a 10-class output layer. Intermediate values in the MRP portion are binary. | 700 | `F49145B9A29193B0927D3CCF5642F3C203479B4399DD25DC1C2ECF6D7F1C67D4` |

All three checkpoints correspond to the **final epoch (700)**, not necessarily the best-performing epoch. The saved optimizer states have a learning rate of 0.001, betas of (0.9, 0.999), epsilon of 1e-8, and weight decay of 0. The retained training implementations use RAdam: the 6-layer MNN uses `torch.optim.RAdam`, while the corrected 6-layer BMNN uses `torch_optimizer.RAdam`. The implementation library used for the 11-layer model cannot be identified from its saved optimizer state alone.

For all three models, the recorded MRP placement configuration is `MX=6`, `MY=6`, and `N=4`. The 6-layer models use right-to-left dataflow with `FLIP_UP=1`, while the 11-layer model uses folded dataflow with `FLIP_UP=0`. These settings describe the relationship between the saved evaluation results and RTL outputs; the placement-generation program is not publicly included.

## Original NNs and Truth-Table Networks

| Model | Correct predictions, original NN | Correct predictions, truth-table network | Matching terminal bits |
| --- | ---: | ---: | ---: |
| [mnn6](evaluation/mnn6/) | 9,473/10,000 (94.73%) | 3,427/10,000 (34.27%) | 53,390/80,000 bits |
| [bmnn6](evaluation/bmnn6/) | 9,240/10,000 (92.40%) | 9,240/10,000 (92.40%) | 80,000/80,000 bits |
| [deep_bmnn11](evaluation/deep_bmnn11/) | 9,229/10,000 (92.29%) | 9,229/10,000 (92.29%) | 80,000/80,000 bits |

For each model, `raw/` contains the classification metrics, predictions, and MRP-terminal bits of the original NN; `truth/` contains the corresponding data for the truth-table network. The file `comparisons/mrp_end_raw_vs_truth.json` records a comparison of terminal bits in the same evaluation order. The original MNN uses continuous-valued intermediate outputs, whereas truth-table generation uses binary inputs and outputs. Therefore, the converted model's 34.27% accuracy must **not** be interpreted as the original MNN's accuracy or as the accuracy of a functionally equivalent implementation.

## Saved RTL Results

| Model | Classification from saved RTL outputs | Agreement with truth-table terminal bits | Evaluation record |
| --- | ---: | ---: | --- |
| 6-layer BMNN | 9,240/10,000 (92.40%) | 80,000/80,000 bits | [rtl_results/bmnn6/evaluation.json](rtl_results/bmnn6/evaluation.json) |
| 11-layer Deep-BMNN | 9,229/10,000 (92.29%) | 80,000/80,000 bits | [rtl_results/deep_bmnn11/evaluation.json](rtl_results/deep_bmnn11/evaluation.json) |

Both RTL prediction CSV files agree with the corresponding truth-table-network prediction CSV files for every test sample. The public `evaluation.json` files retain the saved numerical evaluation results while using paths that resolve within this directory. Although the prediction CSV files include ground-truth labels, the separate output-layer file and evaluation code are not provided. Consequently, the original evaluation procedure cannot be rerun unchanged using this directory alone.

The 11-layer classification record was obtained by retrospectively evaluating 10,000 previously saved XSim outputs. All 80,000 RTL terminal bits and the classification predictions agree with the current truth-table evaluation. However, a separate manifest documenting the complete set of conversion inputs used at the time is no longer available. Agreement with the saved outputs must therefore be distinguished from complete traceability of the original input-generation process. These records do **not** constitute a new XSim run, synthesis or timing evaluation, or measurement on physical hardware.

## Pre-Publication Checklist

Confirm the final manuscript title, citation details, and contact information; the final numerical results reported in the manuscript; the rights to redistribute the models and evaluation materials; and the applicable terms of use. Once finalized, reflect the public repository URL and the scope of available non-public programs in the manuscript's Data availability statement.
