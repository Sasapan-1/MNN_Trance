# Final Epoch Checkpoints

Both files were loaded with PyTorch `weights_only=True` and report `epoch=700`.
They contain model state, optimizer state, and train/test histories.

| Model | File | SHA-256 | Embedded final test accuracy | Stored later inference accuracy |
| --- | --- | --- | ---: | ---: |
| 6-layer binary BMNN | `bmnn_6_epoch700/MNIST_DeepBMNNmodel1v5_6_r_para.pth` | `BB1ADAFD50A730BA1C4E4953E5A5B6DC1309766DAD49FFA5CCE37132D2D6698A` | 91.48% | 91.42% |
| 11-layer Deep-BMNN | `deep_bmnn_11_epoch700/MNIST_DeepBMNNmodel1v5_11_para.pth` | `F49145B9A29193B0927D3CCF5642F3C203479B4399DD25DC1C2ECF6D7F1C67D4` | 92.29% | 92.29% |

Original paths relative to the MNIST project:

- `saved_model/MNIST_DeepBMNNmodel1v5_6_r/MNIST_DeepBMNNmodel1v5_6_r_para.pth`
- `saved_model/mnist11_forward_continue_700/MNIST_DeepBMNNmodel1v5_11_para.pth`

The original model code uses a checkpoint dictionary key named
`MNIST_DeepBMNNmodel1v5`; the files are not bare state dictionaries.
