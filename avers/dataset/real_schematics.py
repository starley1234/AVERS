"""
Real (non-synthetic) Soviet/Russian schematic reference material for the v0.3
Active Learning Loop.

Full write-up (tiers, licensing notes, how to scale up): see
``docs/REAL_SCHEMATICS_SOURCES.md``. A small curated sample (12 images) is
already committed at ``data/reference_schematics/`` for immediate smoke
testing — see ``data/reference_schematics/README.md``.

Design choice: this module deliberately does **not** auto-crawl third-party
websites. Scraping behaviour (robots.txt, rate limits, ToS) varies a lot
across the forums/archives listed here, and bulk-downloading copyrighted
scans should be a conscious, reviewed action taken by a human on the target
(GPU) machine — not something that silently runs in CI or agent automation.

Instead this module gives you:

  - ``REAL_SCHEMATIC_SOURCES``: a structured catalog of where to find more
    material per difficulty tier (mirrors the curated manifest.json).
  - ``load_curated_manifest()``: read the metadata for the committed sample.
  - ``fetch_urls()``: a small, polite, rate-limited downloader for a list of
    *specific* image URLs you already decided to use (copy them from a
    gallery page yourself, or script your own scraper against one source
    and feed the resulting URL list in here).
"""

from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from avers.core.logger import get_logger

logger = get_logger("avers.dataset.real_schematics")

CURATED_SAMPLE_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "reference_schematics"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (compatible; AVERS-research-bot/0.3; "
    "+https://github.com/starley1234/AVERS; internal R&D dataset collection)"
)


@dataclass
class RealSchematicSource:
    """One place to find more real schematics for a given difficulty tier."""

    tier: str  # tier0_gost_symbols | tier1_simple_car | tier2_medium_radio | tier3_complex_truck | tier4_theory
    name: str
    url: str
    description: str
    approx_count: str = "unknown"
    license_note: str = "unclear - verify before redistribution; internal R&D use only"


REAL_SCHEMATIC_SOURCES: List[RealSchematicSource] = [
    # --- tier0: GOST symbol atlases (trivial) ---------------------------- #
    RealSchematicSource(
        tier="tier0_gost_symbols",
        name="opengost.ru - GOST 2.721-74",
        url="https://www.opengost.ru/iso/01_gosty/01080_gost_iso/0108040_gost_iso/1426-gost-2.721-74-eskd.-oboznacheniya-uslovnye-graficheskie-v-shemah.-oboznacheniya-obschego-primeneniya.html",
        description="Полный текст + иллюстрации ГОСТ 2.721-74 (условные обозначения общего применения).",
        approx_count="~150 individual symbol drawings",
        license_note="Государственный стандарт; иллюстрации широко перепечатываются в инженерной литературе.",
    ),
    RealSchematicSource(
        tier="tier0_gost_symbols",
        name="meganorm.ru - GOST 2.709-89 / 2.702-2011 PDFs",
        url="https://meganorm.ru/Data2/1/4293800/4293800211.pdf",
        description="Официальные сканы ГОСТ с примерами условных обозначений проводов/контактов.",
    ),
    # --- tier1: Soviet/Russian car wiring diagrams (simple, closest proxy to small БКС) --- #
    RealSchematicSource(
        tier="tier1_simple_car",
        name="auto-schema.ru",
        url="https://auto-schema.ru/",
        description="Большой архив схем электрооборудования советских/российских авто (ВАЗ, ГАЗ, УАЗ, Москвич) по моделям.",
        approx_count="hundreds of diagrams",
    ),
    RealSchematicSource(
        tier="tier1_simple_car",
        name="avto-elektrika-shema.ru",
        url="https://www.avto-elektrika-shema.ru/",
        description="Схемы электрооборудования по моделям ВАЗ/ГАЗ/Москвич, с расшифровкой позиций.",
        approx_count="hundreds of diagrams",
    ),
    RealSchematicSource(
        tier="tier1_simple_car",
        name="avtorukovodstva.jimdofree.com - Электросхемы авто (EWD)",
        url="https://avtorukovodstva.jimdofree.com/%D1%8D%D0%BB%D0%B5%D0%BA%D1%82%D1%80%D0%BE%D1%81%D1%85%D0%B5%D0%BC%D1%8B/",
        description="Коллекция EWD схем: АЗЛК/Москвич, ВАЗ, ГАЗ (включая довоенные/послевоенные модели).",
        approx_count="dozens of models",
    ),
    RealSchematicSource(
        tier="tier1_simple_car",
        name="vk.ru/@azlk_archive - Схемы электрооборудования Москвич",
        url="https://vk.ru/@azlk_archive-shemy-elektrooborudovaniya-moskvich",
        description="Архив схем Москвич 407/408/412/2140 по годам выпуска.",
        approx_count="~10 models x several sheets",
    ),
    RealSchematicSource(
        tier="tier1_simple_car",
        name="За рулём - исторический архив (zr.ru/archive)",
        url="https://www.zr.ru/archive/zr/1934/12/skhiema-eliektrooborudovaniia-avtomobiliei-gaz",
        description="Сканы схем электрооборудования из довоенных/советских номеров журнала «За рулём» (например, ГАЗ-А/ГАЗ-АА, 1934).",
        approx_count="dozens across archive issues",
    ),
    RealSchematicSource(
        tier="tier1_simple_car",
        name="oktja.ru форум - принципиальные электрические схемы автомобилей",
        url="https://oktja.ru/topic/105213-%D0%BF%D1%80%D0%B8%D0%BD%D1%86%D0%B8%D0%BF%D0%B8%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5-%D1%8D%D0%BB%D0%B5%D0%BA%D1%82%D1%80%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B5-%D1%81%D1%85%D0%B5%D0%BC%D1%8B-%D0%B0%D0%B2%D1%82%D0%BE%D0%BC%D0%BE%D0%B1%D0%B8%D0%BB%D0%B5%D0%B9/",
        description="Форумная ветка с десятками схем (ВАЗ, Москвич, ГАЗ, мотоциклы Ява/ИЖ) в высоком разрешении.",
    ),
    # --- tier2: vintage radio schematics (medium, authentic scan noise) --- #
    RealSchematicSource(
        tier="tier2_medium_radio",
        name="qrz.ru/schemes - каталог схем радиоприёмников",
        url="https://www.qrz.ru/schemes/category/40.html",
        description="Большой структурированный каталог: радиоприёмники, передатчики, измерительная техника (СССР + импорт по ленд-лизу).",
        approx_count="hundreds of schematics",
    ),
    RealSchematicSource(
        tier="tier2_medium_radio",
        name="radiolamp.net",
        url="https://radiolamp.net/news/4-priemniki-na-lampax/",
        description="Ламповые приёмники: принципиальные схемы с описаниями, разные уровни сложности (1-V-1 до супергетеродинов).",
        approx_count="dozens",
    ),
    RealSchematicSource(
        tier="tier2_medium_radio",
        name="radiostorage.net",
        url="https://radiostorage.net/",
        description="Ещё один архив ламповых схем + сопутствующей документации по ремонту.",
    ),
    # --- tier3: trucks/buses/tractors (complex, dense wiring) --- #
    RealSchematicSource(
        tier="tier3_complex_truck",
        name="almarka.ru - КамАЗ электрооборудование",
        url="https://almarka.ru/shema-jelektricheskaja-principialnaja-sistemy-upravlenija-jelektrooborudovaniem-kamaz-5490",
        description="Принципиальные схемы системы управления электрооборудованием современных грузовиков КамАЗ, многолистовые.",
    ),
    RealSchematicSource(
        tier="tier3_complex_truck",
        name="allautoinfo.org - электросхемы коммерческого транспорта",
        url="https://allautoinfo.org/wiring-diagram/",
        description="Схемы грузовиков/спецтехники отечественного и зарубежного производства.",
    ),
    # --- tier4: theory / Э3-Э4-Э6 terminology + academic БКС references --- #
    RealSchematicSource(
        tier="tier4_theory",
        name="Habr - Схемы электрические. Типы схем",
        url="https://habr.com/ru/articles/451158/",
        description="Наглядное объяснение разницы Э1-Э8 (структурная/функциональная/принципиальная/соединений/общая/расположения) с иллюстрациями по ГОСТ 2.702-2011 — полезно для понимания домена перед разметкой реальных БКС.",
    ),
    RealSchematicSource(
        tier="tier4_theory",
        name="sciup.org - Проектирование компонентов БКС ЛА",
        url="https://sciup.org/proektirovanie-komponentov-bortovyh-kabelnyh-setej-s-uchjotom-trebovanij-170200576",
        description="Научная статья про формализацию БКС (Э3 -> Э4 синтез, топологический граф жгутов) - полезный теоретический контекст для graph_synthesis stage, не содержит готовых изображений схем.",
        license_note="Текст статьи для ознакомления; не источник изображений.",
    ),
]


def list_sources(tier: Optional[str] = None) -> List[RealSchematicSource]:
    if tier is None:
        return list(REAL_SCHEMATIC_SOURCES)
    return [s for s in REAL_SCHEMATIC_SOURCES if s.tier == tier]


def load_curated_manifest(root: Path = CURATED_SAMPLE_ROOT) -> dict:
    """Load metadata for the small curated sample committed in the repo."""
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Curated manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def fetch_urls(
    urls: List[str],
    output_dir: Path,
    delay_sec: float = 1.0,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout_sec: float = 15.0,
) -> List[Path]:
    """Download a human-curated list of specific image URLs politely.

    Intentionally requires an explicit URL list rather than crawling a page
    for you - review what you're about to download first. Rate-limits
    requests (``delay_sec`` between downloads) to be a good citizen of the
    (mostly small, volunteer-run) sites in ``REAL_SCHEMATIC_SOURCES``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for i, url in enumerate(urls):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                data = resp.read()
            suffix = Path(url.split("?")[0]).suffix or ".jpg"
            dest = output_dir / f"real_{i:04d}{suffix}"
            dest.write_bytes(data)
            saved.append(dest)
            logger.info(f"Downloaded {url} -> {dest}")
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")

        if i < len(urls) - 1 and delay_sec > 0:
            time.sleep(delay_sec)

    return saved
