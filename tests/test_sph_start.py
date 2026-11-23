import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drake_trajectory_export import PoseInterpolation
from evaluate_sph_spill import DEFAULT_TEMPLATE
from prepare_sph_pouring import PouringConfig, prepare_scene
from sph_start import choose_start


CONFIG = PouringConfig(fill_fraction=.5, particle_radius=.00175)


def motion(angles, *, origin=0):
    times = np.arange(len(angles), dtype=float) * 2
    rotations = Rotation.from_euler("x", np.asarray(angles)[:, None], degrees=True)
    locations = np.zeros((len(times), 3))
    interpolation = PoseInterpolation(times, locations, rotations.as_quat())
    return SimpleNamespace(times=times, interpolation=interpolation, hold_source=False,
                           input_time_origin_s=origin)


def test_selects_sustained_tilt_after_transport_and_pause():
    replay = motion([0, 3, 1, 4, 4, 4, 4, 25, 58, 75, 78, 0], origin=31)
    start, onset, fallback = choose_start(replay, CONFIG, "auto", 2)
    assert fallback is None
    assert 11 < onset < 14
    assert start == pytest.approx(onset - 2, abs=.15)
    assert choose_start(replay, CONFIG, 40., 2)[0] == 9.


def test_missing_unsafe_and_ambiguous_histories_are_not_skipped():
    replay = motion([0, 50, 0, 0, 0, 10, 36, 75, 80])
    assert choose_start(replay, CONFIG, "auto", 2)[0] == 0
    with pytest.raises(ValueError, match="Unsafe SPH start"):
        choose_start(replay, CONFIG, 9., 2)
    assert "saved profile" in choose_start(
        replay, replace(CONFIG, fill_fraction=None), "auto", 2,
    )[2]
    assert "No sustained" in choose_start(motion([0, 6, 8, 0, 4, 0]), CONFIG, "auto", 2)[2]


def test_scene_starts_at_interpolated_pose_without_changing_csv(tmp_path):
    trajectory = tmp_path / "take.csv"
    times = [20., 30., 40.]
    x = np.array([[.4, -.1, .35], [.6, -.1, .35], [.7, -.1, .35]])
    q = Rotation.from_euler("y", [[0], [10], [20]], degrees=True).as_quat()
    with trajectory.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "body", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"))
        writer.writerows((t, "source", *p, r[3], *r[:3]) for t, p, r in zip(times, x, q))
    original = trajectory.read_bytes()
    interpolated = PoseInterpolation(np.asarray(times) - 20., x, q).pose(5.)
    path = prepare_scene(trajectory, tmp_path / "scene", DEFAULT_TEMPLATE, settings=CONFIG,
                         initial_pose=interpolated)
    scene = json.loads(path.read_text())
    assert trajectory.read_bytes() == (path.parent / "source_trajectory.csv").read_bytes() == original
    np.testing.assert_allclose(scene["RigidBodies"][0]["translation"], interpolated[0])
    assert scene["FluidModels"][0]["translation"] == scene["RigidBodies"][0]["translation"]
    fill = scene["initialFill"]
    assert fill["particle_spacing_m"] == .0035
    assert fill["nominal_particle_mass_kg"] == pytest.approx(.0035**3 * 1000)
    assert abs(fill["target_fluid_mass_kg"] - fill["estimated_initialized_mass_kg"]) <= fill["nominal_particle_mass_kg"] / 2


def test_yaw_and_early_translation_kick_are_handled_conservatively():
    yaw = motion([0, 0, 0, 0, 0])
    yaw.interpolation = PoseInterpolation(
        yaw.times, np.zeros((5, 3)),
        Rotation.from_euler("z", [[0], [40], [80], [100], [115]], degrees=True).as_quat(),
    )
    assert choose_start(yaw, CONFIG, "auto", 2)[0] == 0

    replay = motion([0, 0, 0, 0, 0, 0, 4, 25, 56, 80, 80])
    xyz = np.zeros((len(replay.times), 3))
    xyz[1, 0] = 3.
    replay.interpolation = PoseInterpolation(
        replay.times, xyz, replay.interpolation.rotation(replay.times).as_quat(),
    )
    assert "rim" in choose_start(replay, CONFIG, "auto", 2)[2]
