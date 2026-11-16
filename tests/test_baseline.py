"""Official fixation conversion, sealed final partition, and pixel-window features."""

from pathlib import Path

import numpy as np
import pytest

from humangazelab.check_fixations import (
    OFFICIAL_SHA256,
    CheckFixationsUnavailable,
    FinalPartitionSealed,
    assert_open_partition,
    file_sha256,
    official_check_fixations,
)
from humangazelab.image_features import center_antipode, directional_energy
from humangazelab.mit1003 import assign_partition

MFILE = Path("/tmp/checkFixations.m")
ARCHIVE = Path("data/raw/external/mit1003_DATA.zip")


def test_final_partition_is_rejected_before_any_sample_is_read():
    stem = next(f"stem-{index}" for index in range(1000) if assign_partition(f"stem-{index}") == "final")
    with pytest.raises(FinalPartitionSealed):
        assert_open_partition(stem)
    assert assert_open_partition("i05june05_static_street_boston_p1010764") == "train"


def test_directional_energy_responds_to_a_vertical_edge_and_the_antipode_keeps_radius():
    gray = np.zeros((12, 16))
    gray[:, 8:] = 1.0
    energy = directional_energy(gray)
    assert energy["horizontal"][:, 8].mean() > energy["vertical"].mean()
    xs = np.array([2.0, 14.0])
    ys = np.array([3.0, 9.0])
    anti_x, anti_y = center_antipode(xs, ys, 16, 12)
    assert np.allclose(anti_x, np.array([14.0, 2.0]))
    assert np.allclose(anti_y, np.array([9.0, 3.0]))


@pytest.mark.skipif(not MFILE.is_file(), reason="authors' checkFixations.m is not on this machine")
def test_official_detector_matches_a_two_plateau_trace():
    assert file_sha256(MFILE) == OFFICIAL_SHA256
    eye = np.vstack(
        [
            np.tile([100.0, 40.0], (50, 1)),
            np.linspace([100.0, 40.0], [240.0, 40.0], 5, endpoint=False),
            np.tile([240.0, 40.0], (50, 1)),
        ]
    )
    try:
        fixations = official_check_fixations(eye, MFILE)
    except CheckFixationsUnavailable as exc:
        pytest.fail(str(exc))
    assert len(fixations) >= 2
    assert fixations[0]["x"] == pytest.approx(100.0, abs=1.0)
    assert fixations[0]["y"] == pytest.approx(40.0, abs=1.0)
    assert fixations[-1]["x"] == pytest.approx(240.0, abs=1.0)
    assert all(row["duration_ms"] > 0 for row in fixations)


@pytest.mark.skipif(not (MFILE.is_file() and ARCHIVE.is_file()), reason="MIT1003 archive or m-file is absent")
def test_released_training_trace_matches_the_independent_octave_session():
    import io
    import zipfile

    from scipy.io import loadmat

    with zipfile.ZipFile(ARCHIVE) as archive:
        payload = loadmat(
            io.BytesIO(archive.read("DATA/ajs/i05june05_static_street_boston_p1010764.mat")),
            squeeze_me=True,
            struct_as_record=False,
        )
    eye = payload["i05june05_static_street_boston_p1010764"].DATA.eyeData
    fixations = official_check_fixations(eye, MFILE)
    assert fixations[0]["x"] == pytest.approx(516.17721519, abs=1e-6)
    assert fixations[0]["y"] == pytest.approx(373.44000000, abs=1e-6)
    assert fixations[0]["duration_ms"] == pytest.approx(266.66666667, abs=1e-6)
    assert len(fixations) == 10
