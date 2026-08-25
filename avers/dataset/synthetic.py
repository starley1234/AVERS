"""
Синтетическая генерация датасета ГОСТ УГО для RT-DETR / YOLO.

Генерирует изображения схем с размеченными УГО, имитируя реальные сканы.
"""

import random
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import cv2

from avers.dataset.gost_symbols import GOST_SYMBOLS, DRAW_FUNCTIONS, draw_symbol, GOSTSymbol
from avers.core.logger import get_logger

logger = get_logger("avers.dataset.synthetic")


@dataclass
class Annotation:
    """Аннотация объекта."""
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = 1.0
    
    def to_yolo(self, img_w: int, img_h: int) -> str:
        """Конвертировать в YOLO формат."""
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return f"{self.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
    
    def to_coco(self, ann_id: int, image_id: int) -> Dict:
        """Конвертировать в COCO формат."""
        x1, y1, x2, y2 = self.bbox
        return {
            "id": ann_id,
            "image_id": image_id,
            "category_id": self.class_id,
            "bbox": [x1, y1, x2-x1, y2-y1],
            "area": (x2-x1)*(y2-y1),
            "iscrowd": 0,
        }


@dataclass
class SyntheticConfig:
    """Конфиг синтетической генерации."""
    image_size: int = 1024
    background_color: Tuple[int, int, int] = (255, 255, 255)
    
    # Количество объектов
    min_objects: int = 5
    max_objects: int = 25
    
    # Классы для генерации
    classes: List[str] = field(default_factory=lambda: [
        "connector_body", "pin", "junction_dot", "ground", "diode", "resistor"
    ])
    
    # Аугментации
    enable_noise: bool = True
    enable_blur: bool = True
    enable_scan_effects: bool = True
    enable_rotation: bool = False
    enable_wires: bool = True
    
    # Шум
    noise_level: float = 0.05
    blur_kernel: int = 3
    
    # Провода
    wire_thickness: int = 2
    wire_color: Tuple[int, int, int] = (0, 0, 0)
    
    # Разметка
    add_text_labels: bool = True
    text_font_scale: float = 0.5


class SyntheticGenerator:
    """Базовый генератор синтетических изображений."""
    
    def __init__(self, config: Optional[SyntheticConfig] = None):
        self.config = config or SyntheticConfig()
        self.class_name_to_id = {s.class_name: s.class_id for s in GOST_SYMBOLS.values()}
        self.id_to_symbol = GOST_SYMBOLS
    
    def generate_image(self) -> Tuple[np.ndarray, List[Annotation]]:
        """Сгенерировать одно изображение с аннотациями."""
        size = self.config.image_size
        img = np.ones((size, size, 3), dtype=np.uint8) * np.array(self.config.background_color, dtype=np.uint8)
        
        annotations = []
        occupied = []  # занятые области для avoid overlap
        
        num_objects = random.randint(self.config.min_objects, self.config.max_objects)
        
        for _ in range(num_objects):
            class_name = random.choice(self.config.classes)
            symbol = self._get_symbol_by_name(class_name)
            if not symbol:
                continue
            
            # Случайный размер
            w = random.randint(*symbol.width_range)
            h = random.randint(*symbol.height_range)
            
            # Случайная позиция с проверкой перекрытий
            for _attempt in range(20):
                x1 = random.randint(10, size - w - 10)
                y1 = random.randint(10, size - h - 10)
                x2, y2 = x1 + w, y1 + h
                bbox = (x1, y1, x2, y2)
                
                if not self._overlaps(bbox, occupied):
                    # Рисуем
                    draw_symbol(img, symbol, bbox)
                    annotations.append(Annotation(
                        class_id=symbol.class_id,
                        class_name=symbol.class_name,
                        bbox=bbox
                    ))
                    occupied.append(bbox)
                    
                    # Если разъем - добавить пины
                    if symbol.has_pins and class_name == "connector_body":
                        pin_count = random.randint(*symbol.pin_count_range)
                        pin_anns = self._generate_pins(img, bbox, pin_count, occupied)
                        annotations.extend(pin_anns)
                    
                    break
        
        # Провода между компонентами
        if self.config.enable_wires and len(annotations) > 2:
            self._draw_random_wires(img, annotations)
        
        # Текстовые метки
        if self.config.add_text_labels:
            self._add_text_labels(img, annotations)
        
        # Аугментации
        img = self._augment(img)
        
        return img, annotations
    
    def _get_symbol_by_name(self, name: str) -> Optional[GOSTSymbol]:
        for s in GOST_SYMBOLS.values():
            if s.class_name == name:
                return s
        return None
    
    def _overlaps(self, bbox: Tuple[int, int, int, int], occupied: List[Tuple[int, int, int, int]], margin: int = 10) -> bool:
        x1, y1, x2, y2 = bbox
        for ox1, oy1, ox2, oy2 in occupied:
            if not (x2 + margin < ox1 or x1 > ox2 + margin or y2 + margin < oy1 or y1 > oy2 + margin):
                return True
        return False
    
    def _generate_pins(self, img: np.ndarray, connector_bbox: Tuple[int, int, int, int], count: int, occupied: List) -> List[Annotation]:
        """Генерировать пины для разъема."""
        x1, y1, x2, y2 = connector_bbox
        anns = []
        
        # Пины справа от разъема
        pin_spacing = (y2 - y1) // max(count, 1)
        for i in range(count):
            py = y1 + pin_spacing//2 + i*pin_spacing
            px = x2 + 5
            pin_bbox = (px, py-4, px+12, py+4)
            
            # Рисуем пин
            cv2.circle(img, (px+2, py), 3, (0,0,0), -1)
            cv2.line(img, (px+2, py), (px+15, py), (0,0,0), 1)
            
            anns.append(Annotation(
                class_id=self.class_name_to_id.get("pin", 1),
                class_name="pin",
                bbox=pin_bbox
            ))
        
        return anns
    
    def _draw_random_wires(self, img: np.ndarray, annotations: List[Annotation]):
        """Нарисовать случайные провода между компонентами."""
        # Выбираем случайные пары
        connectors = [a for a in annotations if a.class_name in ("connector_body", "pin", "ground")]
        if len(connectors) < 2:
            return
        
        for _ in range(random.randint(2, 6)):
            a1, a2 = random.sample(connectors, 2)
            x1 = (a1.bbox[0] + a1.bbox[2])//2
            y1 = (a1.bbox[1] + a1.bbox[3])//2
            x2 = (a2.bbox[0] + a2.bbox[2])//2
            y2 = (a2.bbox[1] + a2.bbox[3])//2
            
            # Ломаная линия
            if random.random() > 0.5:
                mx = (x1 + x2)//2 + random.randint(-30, 30)
                cv2.line(img, (x1, y1), (mx, y1), self.config.wire_color, self.config.wire_thickness)
                cv2.line(img, (mx, y1), (mx, y2), self.config.wire_color, self.config.wire_thickness)
                cv2.line(img, (mx, y2), (x2, y2), self.config.wire_color, self.config.wire_thickness)
            else:
                cv2.line(img, (x1, y1), (x2, y2), self.config.wire_color, self.config.wire_thickness)
            
            # Иногда добавляем точку соединения
            if random.random() > 0.7:
                jx, jy = (x1+x2)//2, (y1+y2)//2
                cv2.circle(img, (jx, jy), 4, (0,0,0), -1)
                annotations.append(Annotation(
                    class_id=self.class_name_to_id.get("junction_dot", 2),
                    class_name="junction_dot",
                    bbox=(jx-4, jy-4, jx+4, jy+4)
                ))
    
    def _add_text_labels(self, img: np.ndarray, annotations: List[Annotation]):
        """Добавить текстовые метки ГОСТ."""
        labels = ["X1", "X2", "Ш1", "VD1", "R1", "K1", "+27В", "GND", "1", "2", "3", "БПВЛ-0.35"]
        for ann in annotations:
            if ann.class_name in ("connector_body", "diode", "resistor", "relay") and random.random() > 0.5:
                x1, y1, x2, y2 = ann.bbox
                text = random.choice(labels)
                cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    
    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Применить аугментации для имитации скана."""
        # Шум
        if self.config.enable_noise:
            noise = np.random.normal(0, self.config.noise_level*255, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Размытие (имитация скана)
        if self.config.enable_blur and random.random() > 0.5:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        
        # Эффекты скана - неравномерное освещение
        if self.config.enable_scan_effects and random.random() > 0.6:
            # Градиент освещения
            h, w = img.shape[:2]
            gradient = np.zeros((h, w), dtype=np.float32)
            for y in range(h):
                gradient[y, :] = 0.9 + 0.2 * math.sin(y / h * math.pi)
            gradient = np.stack([gradient]*3, axis=2)
            img = np.clip(img.astype(np.float32) * gradient, 0, 255).astype(np.uint8)
        
        # Поворот
        if self.config.enable_rotation and random.random() > 0.8:
            angle = random.uniform(-2, 2)
            M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=(255,255,255))
        
        return img
    
    def generate_dataset(self, num_images: int, output_dir: Path, split: str = "train"):
        """Сгенерировать датасет."""
        output_dir = Path(output_dir)
        images_dir = output_dir / "images" / split
        labels_dir = output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating {num_images} images to {output_dir}")
        
        for i in range(num_images):
            img, anns = self.generate_image()
            
            # Сохранить изображение
            img_path = images_dir / f"{split}_{i:06d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            # Сохранить YOLO аннотации
            label_path = labels_dir / f"{split}_{i:06d}.txt"
            with open(label_path, "w") as f:
                for ann in anns:
                    f.write(ann.to_yolo(img.shape[1], img.shape[0]) + "\n")
            
            if (i+1) % 100 == 0:
                logger.info(f"Generated {i+1}/{num_images}")
        
        # Создать dataset.yaml
        self._create_yaml(output_dir)
        
        logger.info(f"Dataset generated at {output_dir}")
    
    def _create_yaml(self, output_dir: Path):
        """Создать dataset.yaml для YOLO."""
        yaml_content = f"""
# AVERS ГОСТ УГО Dataset
path: {output_dir}
train: images/train
val: images/val
test: images/test

nc: {len(GOST_SYMBOLS)}
names:
"""
        for id, sym in GOST_SYMBOLS.items():
            yaml_content += f"  {id}: {sym.class_name}\n"
        
        with open(output_dir / "dataset.yaml", "w") as f:
            f.write(yaml_content)


class GOSTGenerator(SyntheticGenerator):
    """Расширенный генератор с ГОСТ-специфичными схемами."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config.enable_wires = True
        self.config.add_text_labels = True
    
    def generate_realistic_schematic(self) -> Tuple[np.ndarray, List[Annotation]]:
        """Сгенерировать реалистичную схему БКС."""
        size = self.config.image_size
        img = np.ones((size, size, 3), dtype=np.uint8) * 255
        
        annotations = []
        
        # Размещаем разъемы слева и справа (типично для БКС)
        left_connectors = random.randint(1, 3)
        right_connectors = random.randint(1, 3)
        
        # Левые разъемы
        for i in range(left_connectors):
            y = 100 + i * 250
            x = 50
            w, h = 60, 120
            bbox = (x, y, x+w, y+h)
            
            symbol = GOST_SYMBOLS[0]  # connector_body
            draw_symbol(img, symbol, bbox)
            annotations.append(Annotation(0, "connector_body", bbox))
            
            # Пины
            pin_count = random.randint(3, 8)
            pin_anns = self._generate_pins(img, bbox, pin_count, [])
            annotations.extend(pin_anns)
        
        # Правые разъемы
        for i in range(right_connectors):
            y = 100 + i * 250
            x = size - 110
            w, h = 60, 120
            bbox = (x, y, x+w, y+h)
            draw_symbol(img, GOST_SYMBOLS[0], bbox)
            annotations.append(Annotation(0, "connector_body", bbox))
            
            # Пины слева от разъема
            pin_count = random.randint(3, 8)
            for j in range(pin_count):
                py = y + 15 + j* (h//pin_count)
                px = x - 15
                cv2.circle(img, (px, py), 3, (0,0,0), -1)
                cv2.line(img, (px, py), (px-10, py), (0,0,0), 1)
                annotations.append(Annotation(1, "pin", (px-5, py-5, px+5, py+5)))
        
        # Случайные компоненты в центре
        for _ in range(random.randint(2, 6)):
            cls_name = random.choice(["diode", "resistor", "ground", "relay"])
            sym = self._get_symbol_by_name(cls_name)
            if not sym:
                continue
            w = random.randint(*sym.width_range)
            h = random.randint(*sym.height_range)
            x = random.randint(size//4, 3*size//4 - w)
            y = random.randint(100, size-100 - h)
            bbox = (x, y, x+w, y+h)
            draw_symbol(img, sym, bbox)
            annotations.append(Annotation(sym.class_id, sym.class_name, bbox))
        
        # Провода - соединяем пины
        self._draw_realistic_wires(img, annotations)
        
        # Точки соединения
        for _ in range(random.randint(1, 4)):
            x = random.randint(size//4, 3*size//4)
            y = random.randint(100, size-100)
            cv2.circle(img, (x, y), 4, (0,0,0), -1)
            annotations.append(Annotation(2, "junction_dot", (x-4, y-4, x+4, y+4)))
        
        # Текст
        self._add_text_labels(img, annotations)
        
        img = self._augment(img)
        return img, annotations
    
    def _draw_realistic_wires(self, img: np.ndarray, annotations: List[Annotation]):
        """Рисует реалистичные трассы проводов."""
        pins = [a for a in annotations if a.class_name == "pin"]
        if len(pins) < 2:
            return
        
        # Соединяем случайные пины
        random.shuffle(pins)
        for i in range(0, len(pins)-1, 2):
            p1, p2 = pins[i], pins[i+1]
            x1 = (p1.bbox[0]+p1.bbox[2])//2
            y1 = (p1.bbox[1]+p1.bbox[3])//2
            x2 = (p2.bbox[0]+p2.bbox[2])//2
            y2 = (p2.bbox[1]+p2.bbox[3])//2
            
            # Ортогональная трассировка (как на реальных схемах)
            mid_x = (x1+x2)//2
            cv2.line(img, (x1, y1), (mid_x, y1), (0,0,0), 2)
            cv2.line(img, (mid_x, y1), (mid_x, y2), (0,0,0), 2)
            cv2.line(img, (mid_x, y2), (x2, y2), (0,0,0), 2)


# Алиас для обратной совместимости
GOSTGenerator = GOSTGenerator
