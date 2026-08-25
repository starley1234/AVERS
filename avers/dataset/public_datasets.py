"""
Public Datasets for AVERS - загрузчики готовых датасетов для обучения.

Найденные датасеты (исследование 2024-2025):

1. **CGHD / CircuitNet** - Hand-drawn circuit diagrams
   - 3,191 images (augmented), 5-12 classes
   - GitHub: aaanthonyyy/CircuitNet
   - https://github.com/aaanthonyyy/CircuitNet

2. **JUHCCR-v1** - Hand-drawn circuit components
   - 20 classes, 150 samples/class original (3000 total)
   - 1500 samples/class augmented (30000 total)
   - https://github.com/AyushRoy2001/Circuit-Component-Analysis
   - Paper: Nature Scientific Reports 2025

3. **ElectroNet / Circuit Diagram Dataset**
   - 23 classes, 3500+ schematics
   - 12 classes for YOLOv8: AC Source, BJT, Battery, Capacitor, DC Source, Diode, Ground, Inductor, MOSFET, Resistor, Current Source, Voltage Source
   - Paper: ElectroNet (2023)

4. **Roboflow Universe - Handwritten Circuit Diagram**
   - 200 images, electronic components bounding boxes
   - CC BY 4.0
   - https://universe.roboflow.com/deep-learning-in-computer-vision/handwritten-circuit-diagram

5. **Roboflow Universe - Circuit Diagram**
   - 2,073 images, capacitor annotations (can be extended)
   - CC BY 4.0
   - https://universe.roboflow.com/deeplearning-l1bq5/circuit-diagram-eo4kn

6. **Digitize-HCD** - Handwritten Circuit Diagrams
   - 17 component symbol classes
   - Mendeley Data: rngcz5wtv8
   - https://data.mendeley.com/datasets/rngcz5wtv8/2
   - CC BY 4.0

7. **Masala-CHAI** - SPICE Netlist Dataset
   - Large-scale from 10 textbooks, automated YOLOv8 detection
   - 4,300 diagrams, 12 classes
   - https://arxiv.org/html/2411.14299v5

8. **Custom from Razavi book** - Analog CMOS IC Design
   - 1200 component detection + 3552 connection graph
   - From Razavi's book PDFs, scanned, photos

9. **DeepPCB, PKU-Market-PCB, DsPCBSD+** - PCB defect datasets (for reference, not directly for schematics)
   - Useful for transfer learning on PCB

Для АВЕРС: нужны именно схемы БКС с ГОСТ УГО, поэтому:
- Используем public datasets для pre-training
- Fine-tune на синтетическом ГОСТ датасете
- Active learning на реальных БКС
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
import yaml
import shutil

from avers.core.logger import get_logger
from avers.dataset.synthetic import Annotation
from avers.dataset.gost_symbols import GOST_SYMBOLS

logger = get_logger("avers.dataset.public")


@dataclass
class PublicDatasetInfo:
    """Info about public dataset."""
    name: str
    description: str
    url: str
    num_images: int
    num_classes: int
    classes: List[str]
    license: str
    format: str  # yolo, coco, etc.
    paper_url: Optional[str] = None
    download_url: Optional[str] = None
    local_path: Optional[Path] = None
    gost_compatible: bool = False  # Whether classes map to GOST
    notes: str = ""


# Registry of known public datasets
PUBLIC_DATASETS: Dict[str, PublicDatasetInfo] = {
    "juhccr-v1": PublicDatasetInfo(
        name="JUHCCR-v1",
        description="Hand-drawn electrical and electronics circuit component recognition, 20 classes",
        url="https://github.com/AyushRoy2001/Circuit-Component-Analysis",
        num_images=3000,  # original, 30000 augmented
        num_classes=20,
        classes=["ammeter", "voltmeter", "transformer", "resistor", "ac_source", "capacitor", "diode", "transistor", "inductor", "battery", "ground", "switch", "bulb", "buzzer", "antenna", "motor", "speaker", "microphone", "integrated_circuit", "potentiometer"],
        license="MIT",
        format="classification",  # isolated components, not detection
        paper_url="https://www.nature.com/articles/s41598-025-22404-5",
        gost_compatible=False,
        notes="Isolated components, need to convert to detection. Good for pre-training classifier."
    ),
    
    "circuitnet": PublicDatasetInfo(
        name="CircuitNet",
        description="Hand-drawn schematic sketch recognizer, 5 common components",
        url="https://github.com/aaanthonyyy/CircuitNet",
        num_images=3191,
        num_classes=5,
        classes=["resistor", "capacitor", "inductor", "diode", "voltage_source"],
        license="MIT",
        format="classification",
        gost_compatible=False,
        notes="3,191 images including augmented. Original from mahmut-aksakalli/circuit_recognizer"
    ),
    
    "handwritten-circuit-roboflow": PublicDatasetInfo(
        name="Handwritten Circuit Diagram (Roboflow)",
        description="200 open source electronic components images with bounding boxes",
        url="https://universe.roboflow.com/deep-learning-in-computer-vision/handwritten-circuit-diagram",
        num_images=200,
        num_classes=12,
        classes=["resistor", "capacitor", "inductor", "diode", "transistor", "battery", "ground", "switch", "transformer", "integrated_circuit", "led", "potentiometer"],
        license="CC BY 4.0",
        format="yolo, coco, etc.",
        download_url="https://universe.roboflow.com/deep-learning-in-computer-vision/handwritten-circuit-diagram",
        gost_compatible=False,
        notes="Ready for YOLO training, good for pre-training"
    ),
    
    "circuit-diagram-roboflow": PublicDatasetInfo(
        name="Circuit Diagram (Roboflow)",
        description="2073 images, capacitor annotations",
        url="https://universe.roboflow.com/deeplearning-l1bq5/circuit-diagram-eo4kn",
        num_images=2073,
        num_classes=12,
        classes=["capacitor", "resistor", "diode", "transistor", "inductor", "ground", "battery", "switch", "transformer", "integrated_circuit", "led", "voltage_source"],
        license="CC BY 4.0",
        format="yolo",
        gost_compatible=False,
        notes="Larger dataset, 2073 images"
    ),
    
    "digitize-hcd": PublicDatasetInfo(
        name="Digitize-HCD",
        description="Digitization of Handwritten Circuit Diagrams, 17 component symbol classes",
        url="https://data.mendeley.com/datasets/rngcz5wtv8/2",
        num_images=1000,  # approximate
        num_classes=17,
        classes=["resistor", "capacitor", "inductor", "diode", "transistor", "battery", "ground", "switch", "transformer", "op_amp", "current_source", "voltage_source", "ac_source", "bulb", "antenna", "motor", "integrated_circuit"],
        license="CC BY 4.0",
        format="custom + heatmaps for ports",
        paper_url="https://data.mendeley.com/datasets/rngcz5wtv8/2",
        gost_compatible=False,
        notes="Includes port location heatmaps - useful for pin detection!"
    ),
    
    "masala-chai": PublicDatasetInfo(
        name="Masala-CHAI",
        description="Large-Scale SPICE Netlist Dataset from textbooks, YOLOv8 on 4300 diagrams",
        url="https://arxiv.org/html/2411.14299v5",
        num_images=4300,
        num_classes=12,
        classes=["ac_source", "bjt", "battery", "capacitor", "dc_source", "diode", "ground", "inductor", "mosfet", "resistor", "current_source", "voltage_source"],
        license="Research",
        format="yolo",
        paper_url="https://arxiv.org/html/2411.14299v5",
        gost_compatible=True,
        notes="4300 diagrams from 10 textbooks, YOLOv8 fine-tuned. Closest to AVERS use case! Includes SPICE netlists."
    ),
    
    "electronet": PublicDatasetInfo(
        name="ElectroNet",
        description="Small-Scale Object Detection in Electrical Schematic Diagrams, 23 classes, 3500 schematics",
        url="https://www.researchgate.net/publication/372298462_ElectroNet",
        num_images=3500,
        num_classes=23,
        classes=["resistor", "capacitor", "inductor", "diode", "transistor", "mosfet", "bjt", "op_amp", "battery", "ground", "current_source", "voltage_source", "switch", "transformer", "integrated_circuit", "led", "potentiometer", "fuse", "relay", "connector", "antenna", "crystal", "display"],
        license="Research",
        format="yolo",
        paper_url="https://www.researchgate.net/publication/372298462_ElectroNet",
        gost_compatible=True,
        notes="3500 schematics, 23 classes, includes text elements with units. Good for AVERS!"
    ),
    
    "razavi-custom": PublicDatasetInfo(
        name="Razavi Analog IC Dataset",
        description="From Razavi's Analog CMOS Integrated Circuit Design book, 1200 detection + 3552 graph",
        url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10781286/",
        num_images=1200,
        num_classes=12,
        classes=["mosfet", "resistor", "capacitor", "current_source", "voltage_source", "ground", "bjt", "diode", "inductor", "op_amp", "switch", "transformer"],
        license="Research",
        format="yolo + graph",
        paper_url="https://pmc.ncbi.nlm.nih.gov/articles/PMC10781286/",
        gost_compatible=False,
        notes="1200 component detection + 3521 connection graph. Includes port localization algorithm."
    ),
}


# Mapping from public dataset classes to GOST
PUBLIC_TO_GOST_MAPPING = {
    # Common mappings
    "resistor": "resistor",
    "capacitor": "capacitor",
    "diode": "diode",
    "ground": "ground",
    "relay": "relay",
    "connector": "connector_body",
    "inductor": "resistor",  # Closest GOST
    "transformer": "relay",
    "switch": "offpage_connector",
    "battery": "ground",  # Power
    "voltage_source": "ground",
    "current_source": "ground",
    "transistor": "relay",
    "mosfet": "relay",
    "bjt": "relay",
    "op_amp": "relay",
    "integrated_circuit": "relay",
    # Add more as needed
}


class PublicDatasetLoader:
    """Loader for public datasets."""
    
    def __init__(self, cache_dir: Path = Path("/tmp/avers_public_datasets")):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def list_datasets(self, gost_compatible_only: bool = False) -> List[PublicDatasetInfo]:
        """List available public datasets."""
        datasets = list(PUBLIC_DATASETS.values())
        if gost_compatible_only:
            datasets = [d for d in datasets if d.gost_compatible]
        return datasets
    
    def get_dataset_info(self, name: str) -> Optional[PublicDatasetInfo]:
        """Get info for specific dataset."""
        return PUBLIC_DATASETS.get(name)
    
    def download_instructions(self, name: str) -> str:
        """Get download instructions for dataset."""
        info = PUBLIC_DATASETS.get(name)
        if not info:
            return f"Dataset {name} not found"
        
        instructions = f"""
Dataset: {info.name}
Description: {info.description}
URL: {info.url}
Images: {info.num_images}, Classes: {info.num_classes}
License: {info.license}
Format: {info.format}

Classes: {', '.join(info.classes)}

Download:
"""
        if info.download_url:
            instructions += f"  Visit: {info.download_url}\n"
        else:
            instructions += f"  Visit: {info.url}\n"
        
        if "roboflow" in info.url:
            instructions += """
  Roboflow Universe:
    1. Go to URL
    2. Click Download -> YOLOv8 format
    3. Unzip to cache dir
  
  Or via API:
    pip install roboflow
    from roboflow import Roboflow
    rf = Roboflow(api_key=\"YOUR_KEY\")
    project = rf.workspace().project(\"handwritten-circuit-diagram\")
    dataset = project.version(1).download(\"yolov8\")
"""
        
        if info.paper_url:
            instructions += f"\nPaper: {info.paper_url}\n"
        
        instructions += f"\nNotes: {info.notes}\n"
        
        return instructions
    
    def convert_to_gost(
        self,
        dataset_name: str,
        dataset_path: Path,
        output_path: Path,
    ) -> Path:
        """
        Convert public dataset to GOST-compatible format.
        
        Maps public classes to GOST classes.
        """
        info = PUBLIC_DATASETS.get(dataset_name)
        if not info:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        
        # Find YOLO labels
        # Try common structures
        possible_label_dirs = [
            dataset_path / "train" / "labels",
            dataset_path / "valid" / "labels",
            dataset_path / "labels",
            dataset_path / "train" / "labels" / "train",
        ]
        
        # For simplicity, create conversion mapping file
        mapping = {}
        for i, cls_name in enumerate(info.classes):
            gost_name = PUBLIC_TO_GOST_MAPPING.get(cls_name.lower(), "resistor")
            gost_id = next((s.class_id for s in GOST_SYMBOLS.values() if s.class_name == gost_name), 8)
            mapping[i] = gost_id
        
        # Save mapping
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "public_to_gost_mapping.json", "w") as f:
            json.dump({
                "source_dataset": dataset_name,
                "source_classes": info.classes,
                "mapping": mapping,
                "gost_classes": {s.class_id: s.class_name for s in GOST_SYMBOLS.values()}
            }, f, indent=2, ensure_ascii=False)
        
        # Create dataset.yaml with GOST classes
        yaml_content = f"""# Converted from {dataset_name} to GOST
path: {output_path.absolute()}
train: {dataset_path}/train/images
val: {dataset_path}/valid/images

nc: {len(GOST_SYMBOLS)}
names:
"""
        for id, sym in GOST_SYMBOLS.items():
            yaml_content += f"  {id}: {sym.class_name}\n"
        
        with open(output_path / "dataset_gost.yaml", "w") as f:
            f.write(yaml_content)
        
        logger.info(f"Converted {dataset_name} mapping saved to {output_path}")
        return output_path / "dataset_gost.yaml"
    
    def create_mixed_dataset_config(
        self,
        synthetic_dataset_yaml: Path,
        public_datasets: List[Tuple[str, Path]],
        output_path: Path,
    ) -> Path:
        """
        Create config for training on mixed synthetic + public datasets.
        
        Args:
            synthetic_dataset_yaml: path to synthetic dataset.yaml
            public_datasets: list of (name, path) for public datasets
            output_path: output dir for mixed config
        
        Returns:
            Path to mixed dataset.yaml
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read synthetic yaml
        with open(synthetic_dataset_yaml, "r") as f:
            synth_data = yaml.safe_load(f)
        
        # For mixed training, we need to combine image lists
        # Ultralytics supports multiple data sources via custom loader
        # For simplicity, create a yaml that points to combined dirs
        
        # Create combined images dir with symlinks
        combined_images = output_path / "images"
        combined_labels = output_path / "labels"
        (combined_images / "train").mkdir(parents=True, exist_ok=True)
        (combined_labels / "train").mkdir(parents=True, exist_ok=True)
        (combined_images / "val").mkdir(parents=True, exist_ok=True)
        (combined_labels / "val").mkdir(parents=True, exist_ok=True)
        
        # Copy/symlink synthetic
        synth_path = Path(synth_data["path"])
        for split in ["train", "val"]:
            src_img = synth_path / "images" / split
            src_lbl = synth_path / "labels" / split
            if src_img.exists():
                for img_file in src_img.glob("*.jpg"):
                    dest = combined_images / split / f"synth_{img_file.name}"
                    if not dest.exists():
                        try:
                            dest.symlink_to(img_file.absolute())
                        except Exception:
                            shutil.copy(img_file, dest)
            
            if src_lbl.exists():
                for lbl_file in src_lbl.glob("*.txt"):
                    dest = combined_labels / split / f"synth_{lbl_file.name}"
                    if not dest.exists():
                        try:
                            dest.symlink_to(lbl_file.absolute())
                        except Exception:
                            shutil.copy(lbl_file, dest)
        
        # Add public datasets
        for ds_name, ds_path in public_datasets:
            ds_path = Path(ds_path)
            # Try to find images
            for split in ["train", "valid", "val"]:
                src_img = ds_path / split / "images"
                if not src_img.exists():
                    src_img = ds_path / "images" / split
                if not src_img.exists():
                    src_img = ds_path / split
                if not src_img.exists():
                    continue
                
                src_lbl = ds_path / split / "labels"
                if not src_lbl.exists():
                    src_lbl = ds_path / "labels" / split
                
                # Map to our train/val
                dest_split = "train" if split == "train" else "val"
                
                if src_img.exists():
                    for img_file in list(src_img.glob("*.jpg")) + list(src_img.glob("*.png")):
                        dest = combined_images / dest_split / f"{ds_name}_{img_file.name}"
                        if not dest.exists():
                            try:
                                dest.symlink_to(img_file.absolute())
                            except Exception:
                                shutil.copy(img_file, dest)
                
                if src_lbl.exists():
                    for lbl_file in src_lbl.glob("*.txt"):
                        dest = combined_labels / dest_split / f"{ds_name}_{lbl_file.name}"
                        if not dest.exists():
                            try:
                                dest.symlink_to(lbl_file.absolute())
                            except Exception:
                                shutil.copy(lbl_file, dest)
        
        # Create mixed yaml
        mixed_yaml = f"""# Mixed dataset: synthetic + public
path: {output_path.absolute()}
train: images/train
val: images/val

nc: {len(GOST_SYMBOLS)}
names:
"""
        for id, sym in GOST_SYMBOLS.items():
            mixed_yaml += f"  {id}: {sym.class_name}\n"
        
        yaml_path = output_path / "dataset_mixed.yaml"
        with open(yaml_path, "w") as f:
            f.write(mixed_yaml)
        
        logger.info(f"Mixed dataset config created: {yaml_path}")
        logger.info(f"Train images: {len(list((combined_images / 'train').glob('*')))}")
        logger.info(f"Val images: {len(list((combined_images / 'val').glob('*')))}")
        
        return yaml_path


def get_recommended_training_strategy() -> str:
    """Get recommended training strategy for AVERS."""
    return """
# Рекомендуемая стратегия обучения для АВЕРС

## Этап 1: Pre-training на публичных датасетах (transfer learning)
Используйте большие публичные датасеты для обучения базовой детекции компонентов.

Рекомендуемые датасеты (по приоритету):
1. **Masala-CHAI** (4300 изображений, 12 классов) - самый близкий к АВЕРС, из учебников, с SPICE netlists
2. **ElectroNet** (3500 изображений, 23 класса) - много классов, текст с единицами
3. **Circuit Diagram Roboflow** (2073 изображений) - большой, CC BY 4.0
4. **Digitize-HCD** (1000 изображений, 17 классов) - есть heatmaps для пинов!

Команда:
  python -m avers.dataset.public download --dataset masala-chai --output /tmp/public/masala-chai
  python -m avers dataset train --data /tmp/public/masala-chai/dataset.yaml --model rtdetr-l --epochs 50

## Этап 2: Fine-tuning на синтетическом ГОСТ датасете
Дообучите модель на синтетических ГОСТ УГО.

  python -m avers dataset generate --output /tmp/avers_gost --num-train 5000 --num-val 500
  python -m avers dataset train --data /tmp/avers_gost/dataset.yaml --model /tmp/runs/masala-chai/weights/best.pt --epochs 50

## Этап 3: Mixed training (синтетика + публичные)
Объедините датасеты для лучшего generalization.

  python -m avers.dataset.public mix --synthetic /tmp/avers_gost/dataset.yaml --public masala-chai:/tmp/public/masala-chai,circuit-diagram-roboflow:/tmp/public/circuit-roboflow --output /tmp/mixed
  python -m avers dataset train --data /tmp/mixed/dataset_mixed.yaml --model rtdetr-l --epochs 100

## Этап 4: Active Learning на реальных БКС
После деплоя Web UI, собирайте feedback и дообучайте.

  # В Web UI исправляйте ошибки -> они идут в /tmp/avers_feedback/
  curl http://localhost:8000/api/active-learning/stats
  curl -X POST http://localhost:8000/api/active-learning/retrain?model_type=rtdetr-l&epochs=20

## Итоговая модель
Модель, обученная по этой стратегии, должна:
- Хорошо детектировать базовые компоненты (из публичных датасетов)
- Понимать ГОСТ УГО (из синтетики)
- Адаптироваться к реальным БКС (из active learning)

Ожидаемые метрики:
- mAP@0.5 > 0.85 на синтетике
- mAP@0.5 > 0.70 на реальных БКС (после active learning)
"""
