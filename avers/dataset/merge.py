"""
Merge datasets for Active Learning retraining (v0.3).

Problem this solves
--------------------
``ActiveLearningLoop.export_for_training`` (avers/active_learning/loop.py)
only exports the handful of validator-corrected crops. Training *exclusively*
on that tiny, skewed set risks catastrophic forgetting of classes that simply
haven't been corrected recently. The roadmap item "merge synthetic + feedback
datasets for training" (ROADMAP.md v0.3) asks for a combined dataset instead.

This module merges N YOLO-format datasets (synthetic GOST, accumulated
feedback, and/or already-GOST-converted public datasets — see
``avers/dataset/public_datasets.py``) into a single YOLO dataset, handling:

  - differing ``nc``/``names`` across sources (remapped onto one canonical
    class list, defaulting to ``avers.dataset.gost_symbols.GOST_SYMBOLS``)
  - optional oversampling weight per source (to upweight rare, high-value
    feedback data relative to abundant synthetic data)
  - train/val splits kept separate per source (falls back to using the
    source's ``train`` split for ``val`` if it doesn't define one — this
    matches what ``ActiveLearningLoop.export_for_training`` already does)

Images are **symlinked** (not copied) into the merged directory whenever
possible to avoid duplicating potentially large datasets on disk; copies are
used as a fallback (e.g. cross-device links, Windows without privileges).

Example
-------
::

    from pathlib import Path
    from avers.dataset.merge import DatasetSource, merge_datasets

    sources = [
        DatasetSource("synthetic", Path("/tmp/avers_gost/dataset.yaml"), weight=1.0),
        DatasetSource("feedback", Path("/tmp/avers_feedback_export/dataset_feedback.yaml"), weight=3.0),
    ]
    yaml_path = merge_datasets(sources, Path("/tmp/avers_merged"))
    # python -m avers dataset train --data /tmp/avers_merged/dataset.yaml --model rtdetr-l
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from avers.core.logger import get_logger
from avers.dataset.gost_symbols import GOST_SYMBOLS

logger = get_logger("avers.dataset.merge")


@dataclass
class DatasetSource:
    """One YOLO-format dataset to merge."""

    name: str
    yaml_path: Path
    weight: int = 1  # how many times to replicate this source's samples (oversampling)


def _load_dataset_yaml(yaml_path: Path) -> Dict:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    base = Path(data.get("path", yaml_path.parent))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()

    names = data.get("names", {})
    # ultralytics accepts names as list or {id: name} dict
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    else:
        names = {int(k): v for k, v in names.items()}

    return {
        "path": base,
        "train": data.get("train", "images/train"),
        "val": data.get("val", data.get("train", "images/train")),
        "names": names,
    }


def _default_canonical_names() -> Dict[int, str]:
    return {cid: sym.class_name for cid, sym in GOST_SYMBOLS.items()}


def _build_class_map(source_names: Dict[int, str], canonical: Dict[str, int]) -> Dict[int, int]:
    """Map a source's local class ids onto canonical (merged) class ids.

    Unknown class names are appended to ``canonical`` so no annotations are
    silently dropped.
    """
    mapping = {}
    for local_id, name in source_names.items():
        if name not in canonical:
            canonical[name] = len(canonical)
        mapping[local_id] = canonical[name]
    return mapping


def _link_or_copy(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    try:
        dest.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dest)


def _remap_label_file(src_label: Path, dest_label: Path, class_map: Dict[int, int]) -> None:
    lines_out = []
    with open(src_label, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                local_id = int(parts[0])
            except ValueError:
                continue
            new_id = class_map.get(local_id, local_id)
            lines_out.append(" ".join([str(new_id)] + parts[1:]))
    dest_label.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def merge_datasets(
    sources: List[DatasetSource],
    output_dir: Path,
    canonical_names: Optional[Dict[int, str]] = None,
) -> Path:
    """Merge multiple YOLO datasets into one combined dataset.yaml.

    Returns the path to the written ``dataset.yaml``.
    """
    if not sources:
        raise ValueError("merge_datasets requires at least one DatasetSource")

    output_dir = Path(output_dir)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # canonical class list: start from GOST_SYMBOLS so ids stay stable across
    # runs, then extend with any class names seen only in a given source.
    canonical: Dict[str, int] = {}
    if canonical_names:
        for cid, name in sorted(canonical_names.items()):
            canonical[name] = cid
    else:
        for cid, name in sorted(_default_canonical_names().items()):
            canonical[name] = cid

    manifest = {"sources": [], "total_images": {"train": 0, "val": 0}}

    for source in sources:
        try:
            ds = _load_dataset_yaml(source.yaml_path)
        except FileNotFoundError as e:
            logger.warning(f"Skipping source '{source.name}': {e}")
            continue

        class_map = _build_class_map(ds["names"], canonical)
        weight = max(1, int(source.weight))

        source_counts = {"train": 0, "val": 0}
        for split in ("train", "val"):
            split_rel = ds.get(split) or ds.get("train")
            src_images_dir = (ds["path"] / split_rel) if not Path(split_rel).is_absolute() else Path(split_rel)
            # images/<split> and labels/<split> are siblings by YOLO convention
            src_labels_dir = Path(str(src_images_dir).replace("images", "labels", 1))

            if not src_images_dir.exists():
                continue

            image_files = sorted(
                [p for p in src_images_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
            )
            for img_path in image_files:
                label_path = src_labels_dir / (img_path.stem + ".txt")
                for rep in range(weight):
                    suffix = f"_{rep}" if rep else ""
                    dest_img = output_dir / "images" / split / f"{source.name}_{img_path.stem}{suffix}{img_path.suffix}"
                    dest_label = output_dir / "labels" / split / f"{source.name}_{img_path.stem}{suffix}.txt"

                    _link_or_copy(img_path, dest_img)
                    if label_path.exists():
                        _remap_label_file(label_path, dest_label, class_map)
                    else:
                        dest_label.write_text("", encoding="utf-8")

                    source_counts[split] += 1

        manifest["sources"].append(
            {
                "name": source.name,
                "yaml": str(source.yaml_path),
                "weight": weight,
                "images_train": source_counts["train"],
                "images_val": source_counts["val"],
            }
        )
        manifest["total_images"]["train"] += source_counts["train"]
        manifest["total_images"]["val"] += source_counts["val"]
        logger.info(
            f"Merged source '{source.name}': {source_counts['train']} train, "
            f"{source_counts['val']} val images (weight={weight})"
        )

    # If val is empty, fall back to train (keeps ultralytics happy, matches
    # the convention already used in ActiveLearningLoop.export_for_training).
    if manifest["total_images"]["val"] == 0:
        val_dir_imgs = output_dir / "images" / "val"
        val_dir_lbls = output_dir / "labels" / "val"
        for img_path in (output_dir / "images" / "train").iterdir():
            _link_or_copy(img_path, val_dir_imgs / img_path.name)
        for lbl_path in (output_dir / "labels" / "train").iterdir():
            dest = val_dir_lbls / lbl_path.name
            if not dest.exists():
                shutil.copy2(lbl_path, dest)

    yaml_lines = [
        "# AVERS merged dataset (synthetic + active-learning feedback [+ public])",
        f"path: {output_dir.resolve()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(canonical)}",
        "names:",
    ]
    for name, cid in sorted(canonical.items(), key=lambda kv: kv[1]):
        yaml_lines.append(f"  {cid}: {name}")

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    (output_dir / "merge_manifest.json").write_text(
        __import__("json").dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info(
        f"Merged {len(sources)} sources -> {yaml_path} "
        f"({manifest['total_images']['train']} train / {manifest['total_images']['val']} val, "
        f"{len(canonical)} classes)"
    )
    return yaml_path
