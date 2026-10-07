"""Conservative frame rejection in production wire vectorization."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from avers.core.validators import ProductionPipeline, WireVectorizer


DATA = Path(__file__).resolve().parents[1] / "data"


def test_dashed_frame_does_not_become_wires_but_nearby_bus_does():
    image = np.full((260, 400, 3), 255, dtype=np.uint8)
    for x in range(32, 367, 18):
        cv2.line(image, (x, 24), (min(x + 11, 366), 24), (0, 0, 0), 2)
        cv2.line(image, (x, 235), (min(x + 11, 366), 235), (0, 0, 0), 2)
    for y in range(24, 236, 18):
        cv2.line(image, (32, y), (32, min(y + 11, 235)), (0, 0, 0), 2)
        cv2.line(image, (366, y), (366, min(y + 11, 235)), (0, 0, 0), 2)

    # Solid bus is only ten pixels above the dashed lower edge, in the
    # same margin band; position alone must not suppress it.
    cv2.line(image, (80, 225), (305, 225), (0, 0, 0), 2)
    segments, junctions = WireVectorizer(min_line_length=3).vectorize(image)

    assert any(abs(a.start[1] - 225) <= 2 and abs(a.end[1] - 225) <= 2
               and abs(a.end[0] - a.start[0]) >= 180 for a in segments)
    assert all(not (abs(a.start[1] - 24) <= 2 and abs(a.end[1] - 24) <= 2)
               for a in segments)
    assert all(not (abs(a.start[1] - 235) <= 2 and abs(a.end[1] - 235) <= 2)
               for a in segments)
    assert all(not (abs(a.start[0] - edge) <= 2 and abs(a.end[0] - edge) <= 2)
               for a in segments for edge in (32, 366))
    assert junctions == {point for a in segments for point in (a.start, a.end)}


def test_real_radio_frame_is_reduced_without_removing_solid_lower_bus():
    path = DATA / "reference_schematics/tier2_medium_radio/tube_radio_kub4_01.png"
    image = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    segments, _ = WireVectorizer(min_line_length=3).vectorize(image)

    frame = [a for a in segments if (28 <= a.start[1] <= 45 and 28 <= a.end[1] <= 45)
             or (680 <= a.start[1] <= 699 and 680 <= a.end[1] <= 699)]
    assert len(frame) <= 4  # The baseline had 17 Hough frame segments.
    assert any(650 <= a.start[1] <= 675 and 650 <= a.end[1] <= 675
               and abs(a.end[0] - a.start[0]) >= 200 for a in segments)


def test_demo_still_has_two_separate_nets():
    demo = DATA / "demo_connections.png"
    if not demo.exists():
        pytest.skip("local demo_connections.png is not tracked")
    image = cv2.cvtColor(cv2.imread(str(demo)), cv2.COLOR_BGR2RGB)
    result = ProductionPipeline().run(image)

    assert result.success
    assert len(result.manifest.nets) == 2
