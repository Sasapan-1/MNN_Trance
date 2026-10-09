# A: model through evaluated truth-table MRP network

This is a local public *candidate*, not an approved release. It contains no
Vivado or SystemVerilog loader generation code.
For commands and verification steps, see [PROCEDURE.md](PROCEDURE.md).

## Scope

- Model definitions, MNIST preprocessing, 6/11-layer training and checkpoints.
- Per-node truth-table extraction, port mapping, MRP placement, and neutral
  MRP-network JSON export.
- Raw NN and truth-table-network inference, accuracy, predictions, and
  MRP-end bit comparison.

The executable entry points are `source/train_mnist.py` and
`source/mnist_mrp_pipeline.py`. The two model definitions used by those
entry points live in `source/model_core.py`; truth-table extraction, graph
construction, and MRP placement use `source/mrp_core.py`. These are extracted
from the original sources without changing their computational bodies. Only
the layer classes required by these models remain in the supporting modules.
The legacy all-in-one sources and unrelated model variants are not included.

## Run

Install `requirements.txt` into a compatible Python environment. From this
directory:

```powershell
python source/train_mnist.py --model bmnn_6 --epochs 700 --out-dir training_runs/bmnn_6
python source/train_mnist.py --model deep_bmnn_11 --epochs 700 --out-dir training_runs/deep_bmnn_11
python source/mnist_mrp_pipeline.py select --models mnist6_binary mnist11_forward
python source/mnist_mrp_pipeline.py prepare --models mnist6_binary mnist11_forward --out-dir runs/neutral
```

MNIST is downloaded to the local data directory if absent. Use `--data-dir`,
`--device`, `--batch-size`, and `--max-samples` as needed. `prepare` uses
the bundled final-epoch checkpoints by default. The stored 6-layer setup uses
right-to-left MRP flow; the 11-layer setup uses folded flow. Override with
`--flow-direction` only for a different experiment.

Each `prepare` run writes `raw/`, `truth/`, `comparisons/`, `inputs/`,
`truth_table/neutral_mrp_network.json`, validation, and a manifest. It does
not write `.mem` or `.svh`; those are B-stage artifacts.

## Collected Evidence

`checkpoints/` and `results/` contain the final epoch-700 checkpoints and
stored 10,000-sample NN/TT comparisons. `generated/` contains historical
neutral MRP JSON, validation, and input/output layer snapshots. Use a new
`prepare` run for a portable handoff.

The stored 6-layer inference accuracy is 91.42%, whereas its checkpoint
training history ends at 91.48%. The stored 11-layer accuracy is 92.29%.
Training from scratch is available but does not guarantee identical weights
or accuracy without the original random state and full environment.
No MNIST dataset, license, citation file, or release approval is included.
