"""Regression checks for the bundled epoch-700 checkpoints."""

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))

import mnist_mrp_pipeline as pipeline
import model_core


CASES = (
    (
        "DeepBMNN_6_layer_reverse_binary",
        "bmnn_6_epoch700",
        "MNIST_DeepBMNNmodel1v5_6_r_para.pth",
        [3, 5],
        [[1, 1, 1, 1, 0, 0, 0, 1], [0, 1, 1, 0, 0, 1, 0, 1]],
    ),
    (
        "DeepBMNN_11_layer",
        "deep_bmnn_11_epoch700",
        "MNIST_DeepBMNNmodel1v5_11_para.pth",
        [7, 5],
        [[0, 1, 1, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 0, 1]],
    ),
)


class CheckpointInferenceTest(unittest.TestCase):
    def test_checkpoint_predictions_and_mrp_bits(self):
        sample = torch.linspace(-1, 1, 784).reshape(1, 1, 28, 28)
        images = torch.cat((sample, -sample), dim=0)
        for class_name, folder, filename, predictions, bits in CASES:
            with self.subTest(class_name=class_name):
                checkpoint = torch.load(
                    ROOT / "checkpoints" / folder / filename,
                    map_location="cpu",
                    weights_only=False,
                )
                self.assertEqual(checkpoint["epoch"], 700)
                model = getattr(model_core, class_name)().eval()
                model.load_state_dict(checkpoint["MNIST_DeepBMNNmodel1v5"])
                with torch.no_grad():
                    logits, mrp = pipeline.forward_split(model, images)
                    self.assertTrue(torch.equal(logits, model(images)))
                actual_bits = (pipeline.flatten_layer_output(mrp) > 0).int()
                self.assertEqual(logits.argmax(dim=1).tolist(), predictions)
                self.assertEqual(actual_bits.tolist(), bits)

    def test_archived_io_layers_load(self):
        for _, folder, _, _, _ in CASES:
            with self.subTest(folder=folder):
                path = ROOT / "generated" / folder / "truth_table" / "export_io_layers"
                input_layer = torch.load(
                    path / "input_layer.pt", map_location="cpu", weights_only=False
                ).eval()
                output_layer = torch.load(
                    path / "output_layer.pt", map_location="cpu", weights_only=False
                ).eval()
                with torch.no_grad():
                    self.assertEqual(len(input_layer(torch.zeros(1, 784))), 6)
                    self.assertEqual(
                        list(output_layer(torch.zeros(1, 8)).shape), [1, 10]
                    )


if __name__ == "__main__":
    unittest.main()
