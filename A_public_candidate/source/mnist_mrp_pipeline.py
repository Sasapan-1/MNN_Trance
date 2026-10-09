from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = ROOT.parent / "runs_mnist_mrp"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    depth: int
    class_name: str
    tag: str
    default_checkpoint_path: Path


MODEL_SPECS: dict[str, ModelSpec] = {
    "mnist6_binary": ModelSpec(
        name="mnist6_binary",
        depth=6,
        class_name="DeepBMNN_6_layer_reverse_binary",
        tag="MNIST_DeepBMNNmodel1v5_6_r",
        default_checkpoint_path=ROOT.parent
        / "checkpoints"
        / "bmnn_6_epoch700"
        / "MNIST_DeepBMNNmodel1v5_6_r_para.pth",
    ),
    "mnist11_forward": ModelSpec(
        name="mnist11_forward",
        depth=11,
        class_name="DeepBMNN_11_layer",
        tag="MNIST_DeepBMNNmodel1v5_11",
        default_checkpoint_path=ROOT.parent
        / "checkpoints"
        / "deep_bmnn_11_epoch700"
        / "MNIST_DeepBMNNmodel1v5_11_para.pth",
    ),
}


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: torch.nn.Module
    source_path: Path
    source_kind: str
    checkpoint_epoch: int | None = None


def import_project_modules(*, quiet: bool = True):
    import matplotlib
    matplotlib.use("Agg", force=True)

    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            import model_core as model_defs
            import mrp_core as mnn
    else:
        import model_core as model_defs
        import mrp_core as mnn

    return model_defs, mnn


def selected_specs(names: Iterable[str]) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    for name in names:
        if name == "all":
            out.extend(MODEL_SPECS.values())
            continue
        if name not in MODEL_SPECS:
            raise KeyError(f"Unknown model name: {name}")
        out.append(MODEL_SPECS[name])
    return out


def make_mnist_loader(
    *,
    data_dir: Path,
    split: str,
    batch_size: int,
    max_samples: int | None,
    download: bool,
) -> DataLoader:
    train = split == "train"
    dataset = datasets.MNIST(
        data_dir,
        train=train,
        download=download,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
    )

    if max_samples is not None:
        dataset = Subset(dataset, list(range(max_samples)))

    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def load_model(
    spec: ModelSpec,
    *,
    device: str,
    checkpoint_path: Path | None = None,
    quiet_import: bool = True,
) -> LoadedModel:
    model_defs, _mnn = import_project_modules(
        quiet=quiet_import
    )

    checkpoint_path = checkpoint_path or spec.default_checkpoint_path

    cls = getattr(model_defs, spec.class_name)
    model = cls().to(device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "MNIST_DeepBMNNmodel1v5" in checkpoint:
        state = checkpoint["MNIST_DeepBMNNmodel1v5"]
        epoch = checkpoint.get("epoch")
    elif isinstance(checkpoint, dict) and all(
        isinstance(key, str) for key in checkpoint.keys()
    ):
        state = checkpoint
        epoch = None
    else:
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    model.load_state_dict(state)
    model.eval()
    return LoadedModel(spec, model, checkpoint_path, "checkpoint", epoch)


def tensor_to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu()


def flatten_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return torch.cat([part.view(part.size(0), -1) for part in output], dim=1)
    return output.view(output.size(0), -1)


def clone_layer_output(output: Any) -> Any:
    if isinstance(output, tuple):
        return tuple(part.clone() for part in output)
    if isinstance(output, list):
        return [part.clone() for part in output]
    return output.clone()


def layer_output_shapes(output: Any) -> list[list[int]]:
    if isinstance(output, (tuple, list)):
        return [list(part.shape[1:]) for part in output]
    return [list(output.shape[1:])]


def forward_split(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, Any]:
    layers = list(model.children())
    if len(layers) < 3:
        raise ValueError("Expected an input layer, at least one MRP layer, and output layer")

    y: Any = layers[0](x.view(x.size(0), -1))
    for layer in layers[1:-1]:
        y = layer(*y) if isinstance(y, (tuple, list)) else layer(y)

    mrp_output = clone_layer_output(y)
    logits_input = clone_layer_output(y)
    logits = (
        layers[-1](*logits_input)
        if isinstance(logits_input, (tuple, list))
        else layers[-1](logits_input)
    )
    return logits, mrp_output


def accuracy_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    labels = labels.to(logits.device)
    preds = logits.argmax(dim=1)
    correct = int((preds == labels).sum().item())
    sample_count = int(labels.numel())
    loss = float(F.nll_loss(logits, labels, reduction="sum").item() / sample_count)
    return {
        "sample_count": sample_count,
        "correct": correct,
        "accuracy": correct / sample_count if sample_count else 0.0,
        "nll_loss": loss,
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def model_spec_payload(spec: ModelSpec) -> dict[str, Any]:
    data = asdict(spec)
    data["default_checkpoint_path"] = str(spec.default_checkpoint_path)
    return data


def save_predictions_csv(
    path: Path, *, labels: torch.Tensor, logits: torch.Tensor
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds = logits.argmax(dim=1).cpu()
    labels = labels.cpu()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "prediction", "correct"])
        for i, (label, pred) in enumerate(zip(labels, preds)):
            writer.writerow([i, int(label), int(pred), int(label == pred)])


def bits_to_hex_line(bits: torch.Tensor) -> str:
    bit_list = bits.detach().cpu().int().view(-1).tolist()
    bit_string = "".join(str(int(bit)) for bit in bit_list)
    pad_len = (-len(bit_string)) % 4
    bit_string = ("0" * pad_len) + bit_string
    hex_width = len(bit_string) // 4
    value = int(bit_string, 2) if bit_string else 0
    return f"{value:0{hex_width}X}"


@torch.no_grad()
def run_raw_inference(
    loaded: LoadedModel,
    loader: DataLoader,
    *,
    device: str,
    out_dir: Path | None = None,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    model = loaded.model
    logits_parts: list[torch.Tensor] = []
    mrp_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    shape_meta: list[list[int]] | None = None

    for x, labels in loader:
        x = x.to(device)
        logits, mrp_output = forward_split(model, x)
        if shape_meta is None:
            shape_meta = layer_output_shapes(mrp_output)
        logits_parts.append(tensor_to_cpu(logits))
        mrp_parts.append(tensor_to_cpu(flatten_layer_output(mrp_output)))
        label_parts.append(labels.cpu())

    logits = torch.cat(logits_parts, dim=0)
    mrp_end = torch.cat(mrp_parts, dim=0)
    labels = torch.cat(label_parts, dim=0)
    mrp_bits = (mrp_end > 0).to(torch.int64)
    metrics = accuracy_metrics(logits, labels)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(logits, out_dir / "final_logits.pt")
        torch.save(mrp_end, out_dir / "mrp_end.pt")
        torch.save(mrp_bits, out_dir / "mrp_end_bits.pt")
        torch.save(labels, out_dir / "labels.pt")
        save_predictions_csv(out_dir / "predictions.csv", labels=labels, logits=logits)
        save_json(out_dir / "metrics.json", metrics)
        save_json(
            out_dir / "mrp_end_meta.json",
            {
                "flattened_width": int(mrp_end.size(1)),
                "parts": shape_meta or [],
                "raw_bits_rule": "1 if raw MRP-end value > 0 else 0",
            },
        )

    return {
        "logits": logits,
        "mrp_end": mrp_end,
        "mrp_bits": mrp_bits,
        "labels": labels,
        "metrics": metrics,
    }


def build_truth_graph(
    model: torch.nn.Module,
    *,
    flip: bool,
    flow_direction: str | None,
    quiet: bool,
) -> Any:
    _model_defs, mnn = import_project_modules(
        quiet=True
    )

    def _build() -> Any:
        layers = list(model.children())
        first_layer = layers[0]
        middle_layers = layers[1:-1]
        last_layer = layers[-1]

        collected_layers: dict[int, Any] = {}
        for i, layer in enumerate(middle_layers):
            collected_layers[i] = mnn.save_layer_module_to_dict(layer)

        graph = mnn.visualize_ml_ml_net_advanse(collected_layers, Flip=flip, flow_direction=flow_direction)
        graph.graph["input_Layer"] = first_layer
        graph.graph["output_Layer"] = last_layer
        return mnn.insert_Truth_table_comp(graph)

    if quiet:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                return _build()
    return _build()


def build_mrp_network_from_truth_graph(
    graph: Any,
    *,
    out_dir: Path,
    quiet: bool,
) -> tuple[Any, dict[str, Any]]:
    _model_defs, mnn = import_project_modules(
        quiet=True
    )

    def _build() -> tuple[Any, dict[str, Any]]:
        mlut_inout = 16
        main_nodes = [
            node for node, attr in graph.nodes(data=True) if attr.get("layer") == "main"
        ]
        min_y = min(int(graph.nodes[node]["pos_y_MRLD"]) for node in main_nodes)
        max_y = max(int(graph.nodes[node]["pos_y_MRLD"]) for node in main_nodes)
        max_x = max(int(graph.nodes[node]["pos_x_MRLD"]) for node in main_nodes)
        y_shift = -min_y if min_y < 0 else 0

        mx = max(int(graph.graph["max_x"]), max_x + 1)
        my = max(int(graph.graph["max_layer_items"]), max_y + y_shift + 1)
        switch = int(graph.graph["flip_up"])
        flow_direction = str(graph.graph.get("flow_direction", "fold" if graph.graph.get("Flip") else "left_to_right"))
        n = mlut_inout // 4

        _points, _connect, pin_matrix = mnn.Create_MLuT_Point(mx, my, n, switch)
        mlut_model = mnn.MLuT_network_view_complex(mx, my, n, switch, pin_matrix)
        mrp_model = mnn.MNN_insert_to_MRP_comp(mlut_model, graph, y_shift=y_shift)

        out_dir.mkdir(parents=True, exist_ok=True)
        old_cwd = Path.cwd()
        try:
            os.chdir(out_dir)
            neutral = mnn.export_mrp_network_to_neutral(
                mrp_model,
                out_path=Path("neutral_mrp_network.json"),
                mx=mx,
                my=my,
                n=n,
                flip_up=switch,
                flow_direction=flow_direction,
            )
        finally:
            os.chdir(old_cwd)

        loaded = mnn.NeutralMRPNetwork.load_json(out_dir / "neutral_mrp_network.json")
        errors = mnn.validate_neutral_mrp_network(loaded)

        meta = {
            "neutral_json": str(out_dir / "neutral_mrp_network.json"),
            "validation_errors": errors,
            "mx": loaded.mx,
            "my": loaded.my,
            "n": loaded.n,
            "flip_up": loaded.flip_up,
            "flow_direction": loaded.flow_direction,
            "y_shift": y_shift,
            "mlut_count": len(loaded.mluts),
            "enabled_mlut_count": sum(1 for m in loaded.mluts if m.enabled),
        }
        save_json(out_dir / "validation.json", meta)
        return mrp_model, meta

    if quiet:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                return _build()
    return _build()


def graph_has_negative_mrp_x(graph: Any) -> bool:
    return any(
        int(attr.get("pos_x_MRLD", 0)) < 0
        for _node, attr in graph.nodes(data=True)
        if attr.get("layer") == "main"
    )


def graph_mrp_x_range(graph: Any) -> tuple[int, int]:
    xs = [
        int(attr["pos_x_MRLD"])
        for _node, attr in graph.nodes(data=True)
        if attr.get("layer") == "main"
    ]
    return min(xs), max(xs)


def normalize_folded_mrp_x(graph: Any) -> None:
    layer_num = int(graph.graph["layer_num"])
    max_x = 0
    for node, attr in graph.nodes(data=True):
        if attr.get("layer") != "main":
            continue
        pos_x = int(attr["pos_x"])
        folded_x = min(pos_x, layer_num - 1 - pos_x)
        graph.nodes[node]["pos_x_MRLD"] = folded_x
        max_x = max(max_x, folded_x + 1)
    graph.graph["max_x"] = max_x


def truth_middle_activate_direct(graph: Any, input_bits: torch.Tensor, device: str) -> Any:
    current = input_bits.to(device).int().view(-1)
    final_parts: list[torch.Tensor] = []
    layer_indices = sorted(
        {
            int(attr["pos_x"])
            for _node, attr in graph.nodes(data=True)
            if attr.get("layer") == "main"
        }
    )

    for layer_index in layer_indices:
        nodes = [
            node
            for node, attr in graph.nodes(data=True)
            if attr.get("layer") == "main" and int(attr["pos_x"]) == layer_index
        ]
        nodes.sort(key=lambda node: int(graph.nodes[node]["pos_y"]))

        start = 0
        parts = []
        for node in nodes:
            input_size = len(graph.nodes[node]["input_port"])
            node_bits = current[start : start + input_size]
            if int(node_bits.numel()) != input_size:
                raise ValueError(
                    f"Not enough bits for truth-table node {node}: "
                    f"expected {input_size}, got {int(node_bits.numel())}"
                )

            bit_string = "".join(str(int(bit)) for bit in node_bits.cpu())
            input_index = int(bit_string, 2) if bit_string else 0
            table_row = graph.nodes[node]["Truth_Table"][input_index]
            if isinstance(table_row, torch.Tensor):
                out = table_row.to(device).view(-1).int()
            else:
                out = torch.tensor(table_row, dtype=torch.int, device=device).view(-1)
            out = torch.where(out > 0, 1, 0).int()
            parts.append(out)
            start += input_size

        current = torch.cat(parts, dim=0) if parts else current
        final_parts = parts

    return tuple(part.float() for part in final_parts)


@torch.no_grad()
def run_truth_inference(
    graph: Any,
    loader: DataLoader,
    *,
    device: str,
    out_dir: Path | None = None,
) -> dict[str, torch.Tensor | dict[str, Any]]:
    _model_defs, mnn = import_project_modules(
        quiet=True
    )

    input_layer = graph.graph["input_Layer"]
    output_layer = graph.graph["output_Layer"]
    input_layer.eval()
    output_layer.eval()

    logits_parts: list[torch.Tensor] = []
    mrp_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    shape_meta: list[list[int]] | None = None

    for x, labels in loader:
        x = x.to(device)
        for sample, label in zip(x, labels):
            sample = sample.unsqueeze(0)
            first = input_layer(sample.view(1, -1))
            first_flat = flatten_layer_output(first)
            first_bits = torch.where(first_flat > 0, 1, 0).squeeze(0).int()

            mrp_tuple = truth_middle_activate_direct(graph, first_bits, device)
            if shape_meta is None:
                shape_meta = layer_output_shapes(mrp_tuple)

            mrp_batch = tuple(
                part.unsqueeze(0) if part.dim() == 1 else part for part in mrp_tuple
            )
            mrp_flat = flatten_layer_output(mrp_batch).clone().int()
            logits_input = tuple(part.clone() for part in mrp_batch)
            logits = output_layer(*logits_input)

            logits_parts.append(tensor_to_cpu(logits))
            mrp_parts.append(tensor_to_cpu(mrp_flat))
            label_parts.append(label.view(1).cpu())

    logits = torch.cat(logits_parts, dim=0)
    mrp_end = torch.cat(mrp_parts, dim=0).to(torch.int64)
    labels = torch.cat(label_parts, dim=0).to(torch.long)
    metrics = accuracy_metrics(logits, labels)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(logits, out_dir / "final_logits.pt")
        torch.save(mrp_end, out_dir / "mrp_end_bits.pt")
        torch.save(labels, out_dir / "labels.pt")
        save_predictions_csv(out_dir / "predictions.csv", labels=labels, logits=logits)
        save_json(out_dir / "metrics.json", metrics)
        save_json(
            out_dir / "mrp_end_meta.json",
            {
                "flattened_width": int(mrp_end.size(1)),
                "parts": shape_meta or [],
                "bits": "truth-table MRP-end output",
            },
        )

    return {
        "logits": logits,
        "mrp_bits": mrp_end,
        "labels": labels,
        "metrics": metrics,
    }


@torch.no_grad()
def export_input_vectors(
    graph: Any,
    loader: DataLoader,
    *,
    device: str,
    out_dir: Path,
    split: str,
) -> dict[str, Any]:
    input_layer = graph.graph["input_Layer"]
    input_layer.eval()

    bit_rows: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    for x, labels in loader:
        x = x.to(device)
        first = input_layer(x.view(x.size(0), -1))
        bits = torch.where(flatten_layer_output(first) > 0, 1, 0).int().cpu()
        bit_rows.append(bits)
        labels_all.append(labels.cpu())

    bits_all = torch.cat(bit_rows, dim=0)
    labels = torch.cat(labels_all, dim=0).to(torch.long)
    bit_width = int(bits_all.size(1))

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(bits_all, out_dir / "input_vectors.pt")
    torch.save(labels, out_dir / "labels.pt")

    labels_path = out_dir / "labels.txt"
    with labels_path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{int(label)}\n")

    meta = {
        "format": "mrp_input_vectors",
        "version": "1.0",
        "source": "MNIST",
        "split": split,
        "sample_count": int(bits_all.size(0)),
        "bit_width": bit_width,
        "bit_order": "MSB-first from flattened input layer outputs",
        "flow_direction": str(graph.graph.get("flow_direction", "left_to_right")),
        "external_input_side": "right" if graph.graph.get("flow_direction") == "right_to_left" else "left",
        "shuffle": False,
        "normalization": {"mean": 0.1307, "std": 0.3081},
        "files": {
            "labels": labels_path.name,
            "pt": "input_vectors.pt",
        },
    }
    save_json(out_dir / "input_meta.json", meta)
    return meta


def compare_bit_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_name: str,
    right_name: str,
    max_examples: int = 20,
) -> dict[str, Any]:
    left = left.detach().cpu().int()
    right = right.detach().cpu().int()
    sample_count = min(int(left.size(0)), int(right.size(0)))
    width = min(int(left.size(1)), int(right.size(1)))
    left = left[:sample_count, :width]
    right = right[:sample_count, :width]

    equal = left == right
    total = int(equal.numel())
    matches = int(equal.sum().item())
    mismatch_positions = (equal == 0).nonzero(as_tuple=False)

    examples = []
    for pos in mismatch_positions[:max_examples]:
        examples.append(
            {
                "sample": int(pos[0]),
                "bit": int(pos[1]),
                left_name: int(left[pos[0], pos[1]]),
                right_name: int(right[pos[0], pos[1]]),
            }
        )

    return {
        "left": left_name,
        "right": right_name,
        "sample_count": sample_count,
        "width": width,
        "match_count": matches,
        "total_bits": total,
        "match_ratio": matches / total if total else 0.0,
        "all_equal": matches == total,
        "mismatch_count": total - matches,
        "first_mismatches": examples,
    }


def command_select(args: argparse.Namespace) -> None:
    specs = selected_specs(args.models)
    loader = make_mnist_loader(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        download=args.download,
    )

    candidates = [(spec, spec.default_checkpoint_path) for spec in specs]

    results = []
    for spec, checkpoint_path in candidates:
        try:
            loaded = load_model(
                spec,
                device=args.device,
                checkpoint_path=checkpoint_path,
                quiet_import=not args.verbose,
            )
            raw = run_raw_inference(loaded, loader, device=args.device, out_dir=None)
            metrics = raw["metrics"]
            results.append(
                {
                    "model": spec.name,
                    "depth": spec.depth,
                    "class_name": spec.class_name,
                    "checkpoint_path": str(checkpoint_path),
                    "source_kind": loaded.source_kind,
                    "epoch": loaded.checkpoint_epoch,
                    "metrics": metrics,
                    "status": "ok",
                }
            )
            print(
                f"{spec.name}: {metrics['accuracy']:.6f} "
                f"({metrics['correct']}/{metrics['sample_count']}) {checkpoint_path}"
            )
        except Exception as exc:
            results.append(
                {
                    "model": spec.name,
                    "depth": spec.depth,
                    "checkpoint_path": str(checkpoint_path),
                    "status": "error",
                    "error": repr(exc),
                }
            )
            print(f"{spec.name}: ERROR {checkpoint_path}: {exc}")

    best_by_model: dict[str, dict[str, Any]] = {}
    best_by_depth: dict[str, dict[str, Any]] = {}
    for result in results:
        if result.get("status") != "ok":
            continue
        model_name = str(result["model"])
        current = best_by_model.get(model_name)
        if current is None or result["metrics"]["accuracy"] > current["metrics"]["accuracy"]:
            best_by_model[model_name] = result
        depth_key = str(result["depth"])
        current_depth = best_by_depth.get(depth_key)
        if (
            current_depth is None
            or result["metrics"]["accuracy"] > current_depth["metrics"]["accuracy"]
        ):
            best_by_depth[depth_key] = result

    payload = {
        "split": args.split,
        "sample_count_limit": args.max_samples,
        "batch_size": args.batch_size,
        "shuffle": False,
        "results": results,
        "best_by_model": best_by_model,
        "best_by_depth": best_by_depth,
    }
    save_json(args.out, payload)
    print(f"saved selection: {args.out}")


def load_selection_checkpoint(selection_path: Path, model_name: str) -> Path | None:
    if not selection_path.exists():
        return None
    data = json.loads(selection_path.read_text(encoding="utf-8"))
    model_result = data.get("best_by_model", {}).get(model_name)
    if not model_result:
        return None
    return Path(model_result["checkpoint_path"])


def prepare_one_model(
    spec: ModelSpec,
    args: argparse.Namespace,
    *,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    run_dir = args.out_dir / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_model(
        spec,
        device=args.device,
        checkpoint_path=checkpoint_path,
        quiet_import=not args.verbose,
    )
    loader = make_mnist_loader(
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        download=args.download,
    )

    dataset_dir = run_dir / "dataset"
    raw_dir = run_dir / "raw"
    truth_dir = run_dir / "truth"
    truth_table_dir = run_dir / "truth_table"
    inputs_dir = run_dir / "inputs"
    comparisons_dir = run_dir / "comparisons"

    print(f"[{spec.name}] raw inference")
    raw = run_raw_inference(loaded, loader, device=args.device, out_dir=raw_dir)

    labels = raw["labels"]
    dataset_dir.mkdir(parents=True, exist_ok=True)
    torch.save(labels, dataset_dir / "labels.pt")
    torch.save(torch.arange(int(labels.numel())), dataset_dir / "sample_indices.pt")
    with (dataset_dir / "labels.txt").open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{int(label)}\n")

    print(f"[{spec.name}] truth table graph")
    flow_direction = args.flow_direction or (
        "right_to_left" if spec.name == "mnist6_binary" else "fold"
    )
    truth_graph = build_truth_graph(
        loaded.model,
        flip=flow_direction == "fold",
        flow_direction=flow_direction,
        quiet=not args.verbose,
    )
    actual_flip = flow_direction == "fold"
    if actual_flip and graph_has_negative_mrp_x(truth_graph):
        print(
            f"[{spec.name}] folded MRP x range {graph_mrp_x_range(truth_graph)}; "
            "normalizing even-layer folded coordinates"
        )
        normalize_folded_mrp_x(truth_graph)

    print(f"[{spec.name}] MRP neutral network export")
    _mrp_model, mrp_meta = build_mrp_network_from_truth_graph(
        truth_graph, out_dir=truth_table_dir, quiet=not args.verbose
    )

    print(f"[{spec.name}] input vector export")
    input_meta = export_input_vectors(
        truth_graph,
        loader,
        device=args.device,
        out_dir=inputs_dir,
        split=args.split,
    )

    print(f"[{spec.name}] truth-table inference")
    truth = run_truth_inference(
        truth_graph,
        loader,
        device=args.device,
        out_dir=truth_dir,
    )

    comparisons_dir.mkdir(parents=True, exist_ok=True)
    raw_truth_compare = compare_bit_tensors(
        raw["mrp_bits"],
        truth["mrp_bits"],
        left_name="raw_binarized",
        right_name="truth_table",
    )
    save_json(comparisons_dir / "mrp_end_raw_vs_truth.json", raw_truth_compare)

    manifest = {
        "model": model_spec_payload(spec),
        "device": args.device,
        "split": args.split,
        "batch_size": args.batch_size,
        "max_samples": args.max_samples,
        "shuffle": False,
        "source": {
            "kind": loaded.source_kind,
            "path": str(loaded.source_path),
            "epoch": loaded.checkpoint_epoch,
        },
        "truth_graph": {
            "flip_requested": flow_direction == "fold",
            "flip_used": actual_flip,
            "flow_direction": flow_direction,
            "mrp_x_range": graph_mrp_x_range(truth_graph),
        },
        "raw_metrics": raw["metrics"],
        "truth_metrics": truth["metrics"],
        "mrp_export": mrp_meta,
        "input_export": input_meta,
        "comparisons": {
            "raw_vs_truth": raw_truth_compare,
        },
        "mrp_output_width": int(truth["mrp_bits"].size(1)),
    }
    save_json(run_dir / "manifest.json", manifest)
    print(f"[{spec.name}] done: {run_dir}")
    return manifest


def command_prepare(args: argparse.Namespace) -> None:
    specs = selected_specs(args.models)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for spec in specs:
        checkpoint_path = args.checkpoint
        if checkpoint_path is None and args.selection_json is not None:
            checkpoint_path = load_selection_checkpoint(args.selection_json, spec.name)
        manifests[spec.name] = prepare_one_model(
            spec, args, checkpoint_path=checkpoint_path
        )
    save_json(args.out_dir / "manifest.json", manifests)
    print(f"saved run manifest: {args.out_dir / 'manifest.json'}")


def add_common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=ROOT.parent / "data")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--verbose", action="store_true")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MNIST 6/11-layer truth-table and neutral MRP evaluation pipeline."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="Evaluate trained candidates for model choice.")
    select.add_argument("--models", nargs="+", default=["mnist6_binary", "mnist11_forward"], choices=["all", *MODEL_SPECS])
    select.add_argument("--out", type=Path, default=DEFAULT_RUN_ROOT / "selection.json")
    add_common_data_args(select)
    select.set_defaults(func=command_select)

    prepare = sub.add_parser(
        "prepare",
        help="Generate raw/truth outputs, a neutral MRP network, and input tensors.",
    )
    prepare.add_argument("--models", nargs="+", default=["mnist6_binary", "mnist11_forward"], choices=["all", *MODEL_SPECS])
    prepare.add_argument("--out-dir", type=Path, default=DEFAULT_RUN_ROOT / "latest")
    prepare.add_argument("--checkpoint", type=Path, default=None)
    prepare.add_argument("--selection-json", type=Path, default=None)
    prepare.add_argument(
        "--flow-direction",
        choices=("fold", "left_to_right", "right_to_left"),
        help="Defaults to right_to_left for the 6-layer model and fold for the 11-layer model.",
    )
    add_common_data_args(prepare)
    prepare.set_defaults(func=command_prepare)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


