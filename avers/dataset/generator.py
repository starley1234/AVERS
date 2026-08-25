"""
Генератор сложных схем - компоновщик для реалистичных БКС.

Создает большие схемы A2x6 формата с множеством компонентов.
"""

import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2

from avers.dataset.synthetic import Annotation, SyntheticConfig, GOSTGenerator
from avers.dataset.gost_symbols import GOST_SYMBOLS, draw_symbol
from avers.core.logger import get_logger

logger = get_logger("avers.dataset.generator")


class SchematicComposer:
    """Компоновщик больших схем БКС."""
    
    def __init__(self, config: Optional[SyntheticConfig] = None):
        self.config = config or SyntheticConfig()
        self.generator = GOSTGenerator(config)
    
    def compose_a2x6(self, width: int = 14000, height: int = 3500) -> Tuple[np.ndarray, List[Annotation]]:
        """Создать схему формата A2x6 (14000x3500)."""
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        annotations = []
        
        # Сетка для размещения
        cols = width // 1000
        rows = height // 500
        
        # Разъемы по краям
        # Левая сторона - входные разъемы
        for i in range(random.randint(3, 8)):
            x = 100
            y = 100 + i * 400 + random.randint(-50, 50)
            w, h = 80, random.randint(100, 250)
            bbox = (x, y, x+w, y+h)
            
            draw_symbol(img, GOST_SYMBOLS[0], bbox)
            annotations.append(Annotation(0, "connector_body", bbox))
            
            # Пины
            pin_count = (h // 20)
            for j in range(pin_count):
                py = y + 15 + j*20
                px = x + w + 5
                cv2.circle(img, (px, py), 3, (0,0,0), -1)
                annotations.append(Annotation(1, "pin", (px-4, py-4, px+4, py+4)))
        
        # Правая сторона - выходные
        for i in range(random.randint(3, 8)):
            x = width - 180
            y = 100 + i * 400 + random.randint(-50, 50)
            w, h = 80, random.randint(100, 250)
            bbox = (x, y, x+w, y+h)
            draw_symbol(img, GOST_SYMBOLS[0], bbox)
            annotations.append(Annotation(0, "connector_body", bbox))
        
        # Центральные компоненты - реле, диоды
        for _ in range(random.randint(10, 30)):
            cls_id = random.choice([6, 7, 8, 3])  # diode, relay, resistor, ground
            sym = GOST_SYMBOLS[cls_id]
            w = random.randint(*sym.width_range)
            h = random.randint(*sym.height_range)
            x = random.randint(width//6, 5*width//6 - w)
            y = random.randint(50, height-50 - h)
            bbox = (x, y, x+w, y+h)
            draw_symbol(img, sym, bbox)
            annotations.append(Annotation(sym.class_id, sym.class_name, bbox))
        
        # Провода - ортогональная трассировка
        self._draw_bus_wires(img, annotations, width, height)
        
        # Точки соединения
        for _ in range(random.randint(20, 60)):
            x = random.randint(200, width-200)
            y = random.randint(50, height-50)
            # Проверяем что рядом есть провода (упрощенно - всегда добавляем)
            cv2.circle(img, (x, y), 4, (0,0,0), -1)
            annotations.append(Annotation(2, "junction_dot", (x-4, y-4, x+4, y+4)))
        
        # Текстовые метки
        labels = [f"X{i}" for i in range(1, 20)] + [f"Ш{i}" for i in range(1, 10)] + \
                 [f"VD{i}" for i in range(1, 15)] + [f"R{i}" for i in range(1, 15)] + \
                 ["+27В", "-12В", "GND", "БПВЛ-0.35", "МГТФ-0.2"]
        
        for ann in annotations:
            if random.random() > 0.7 and ann.class_name in ("connector_body", "diode", "resistor", "relay"):
                x1, y1, _, _ = ann.bbox
                text = random.choice(labels)
                cv2.putText(img, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        
        # Эффекты скана для большого изображения
        img = self._add_scan_artifacts(img)
        
        return img, annotations
    
    def _draw_bus_wires(self, img: np.ndarray, annotations: List[Annotation], width: int, height: int):
        """Рисует шины и жгуты проводов."""
        # Горизонтальные шины
        for _ in range(random.randint(3, 8)):
            y = random.randint(100, height-100)
            x1 = 200
            x2 = width - 200
            # Шина - утолщенная линия
            cv2.line(img, (x1, y), (x2, y), (0,0,0), 3)
        
        # Вертикальные соединения
        for _ in range(random.randint(10, 25)):
            x = random.randint(300, width-300)
            y1 = random.randint(100, height-200)
            y2 = y1 + random.randint(50, 300)
            cv2.line(img, (x, y1), (x, y2), (0,0,0), 2)
    
    def _add_scan_artifacts(self, img: np.ndarray) -> np.ndarray:
        """Добавляет артефакты сканирования для реализма."""
        # Легкий шум
        noise = np.random.normal(0, 8, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Затемнение по краям (виньетка)
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
        max_dist = np.sqrt((w/2)**2 + (h/2)**2)
        vignette = 1 - 0.15 * (dist_from_center / max_dist)
        vignette = np.stack([vignette]*3, axis=2)
        img = np.clip(img.astype(np.float32) * vignette, 0, 255).astype(np.uint8)
        
        return img
    
    def generate_batch(self, num_images: int, output_dir: Path, image_size: Tuple[int, int] = (1024, 1024)):
        """Генерировать батч изображений разного размера."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(num_images):
            if i % 5 == 0 and image_size[0] >= 2000:
                # Каждое 5е - большая схема
                img, anns = self.compose_a2x6(width=4000, height=1000)
            else:
                img, anns = self.generator.generate_realistic_schematic()
            
            # Ресайз если нужно
            if img.shape[1] != image_size[0] or img.shape[0] != image_size[1]:
                img = cv2.resize(img, image_size)
                # Пересчет bbox
                scale_x = image_size[0] / img.shape[1]
                scale_y = image_size[1] / img.shape[0]
                # Упрощенно - не пересчитываем для демо
            
            img_path = output_dir / f"schematic_{i:06d}.jpg"
            cv2.imwrite(str(img_path), img)
            
            logger.info(f"Generated {i+1}/{num_images}: {img_path}")


def generate_full_dataset(
    output_dir: str = "/tmp/avers_dataset",
    num_train: int = 1000,
    num_val: int = 200,
    num_test: int = 100,
    image_size: int = 1024,
):
    """Полный пайплайн генерации датасета."""
    output_dir = Path(output_dir)
    
    config = SyntheticConfig(
        image_size=image_size,
        min_objects=8,
        max_objects=30,
        enable_wires=True,
        enable_noise=True,
        enable_scan_effects=True,
    )
    
    generator = GOSTGenerator(config)
    
    # Train
    logger.info(f"Generating train set: {num_train} images")
    generator.generate_dataset(num_train, output_dir, "train")
    
    # Val
    logger.info(f"Generating val set: {num_val} images")
    generator.generate_dataset(num_val, output_dir, "val")
    
    # Test
    logger.info(f"Generating test set: {num_test} images")
    generator.generate_dataset(num_test, output_dir, "test")
    
    # Дополнительно: большие схемы A2x6
    composer = SchematicComposer(config)
    large_dir = output_dir / "images" / "large"
    large_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating large A2x6 schematics")
    for i in range(20):
        img, anns = composer.compose_a2x6(width=4000, height=1000)
        cv2.imwrite(str(large_dir / f"large_{i:04d}.jpg"), img)
    
    logger.info(f"Full dataset ready at {output_dir}")
    return output_dir
