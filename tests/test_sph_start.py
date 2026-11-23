import csv
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from drake_trajectory_export import PoseInterpolation, TrajectoryRecorder
from evaluate_sph_spill import evaluate
from prepare_sph_pouring import PouringConfig
from sph_start import RestartThresholds, choose_start


CONFIG = PouringConfig(fill_fraction=.5, particle_radius=.00175)


def motion(*, pause_start=2., pour_start=4., tilt=0., jitter=False, origin=0.):
    times = np.linspace(0., 7., 701)
    xyz = np.tile([.4, -.1, .35], (len(times), 1))
    xyz[:, 0] += .09 * np.minimum(times, pause_start)
    angles = tilt + np.clip(40 * (times - pour_start), 0, 75)
    if jitter:
        # Shallow tracking oscillation with large sample-to-sample derivatives.
        xyz[:, 1] += .0003 * np.cos(2 * np.pi * times / .04)
        angles += .15 * np.cos(2 * np.pi * times / .04)
    q = Rotation.from_euler("x", angles[:, None], degrees=True).as_quat()
    return SimpleNamespace(times=times, interpolation=PoseInterpolation(times, xyz, q),
                           hold_source=False, input_time_origin_s=origin)


@pytest.mark.parametrize("jitter", [False, True])
def test_selects_final_sustained_pause_after_transport(jitter):
    replay = motion(jitter=jitter, origin=31.)
    selection = choose_start(replay, CONFIG, "auto", 2)
    assert selection.fallback_reason is None
    assert selection.onset_s == pytest.approx(4., abs=.05)
    assert selection.elapsed_s == pytest.approx(2.08, abs=.06)
    assert selection.interval_s[1] == pytest.approx(3.9, abs=.08)
    assert all(check["passed"] for check in selection.checks.values())
    assert choose_start(replay, CONFIG, 33.2, 2).elapsed_s == pytest.approx(2.2)


def test_falls_back_without_sustained_pause():
    replay = motion(pause_start=3.65)
    selection = choose_start(replay, CONFIG, "auto", 2)
    assert selection.elapsed_s == 0
    assert not selection.checks["stationary_duration_s"]["passed"]
    with pytest.raises(ValueError, match="Unsafe SPH start"):
        choose_start(replay, CONFIG, 3.8, 2)


def test_falls_back_on_tilt_and_insufficient_headroom():
    tilted = choose_start(motion(tilt=16), CONFIG, "auto", 2)
    assert tilted.elapsed_s == 0
    assert not tilted.checks["tilt_deg"]["passed"]
    full_config = replace(CONFIG, fill_fraction=.90, particle_radius=.001)
    high_fill = choose_start(motion(tilt=8), full_config, "auto", 2)
    assert high_fill.elapsed_s == 0
    assert not high_fill.checks["rim_headroom_m"]["passed"]


def test_falls_back_for_immediate_pour_and_missing_fill_or_onset():
    immediate = choose_start(motion(pause_start=0., pour_start=0.), CONFIG, "auto", 2)
    assert immediate.elapsed_s == immediate.onset_s == 0
    assert immediate.fallback_reason
    assert "saved profile" in choose_start(
        motion(), replace(CONFIG, fill_fraction=None), "auto", 2,
    ).fallback_reason
    assert "No sustained" in choose_start(motion(pour_start=7.), CONFIG, "auto", 2).fallback_reason


def test_does_not_use_an_old_pause_during_continued_transport():
    replay = motion(pour_start=5.5)
    xyz = replay.interpolation.translation(replay.times)
    xyz[:, 0] += .1 * np.maximum(0., replay.times - 3.5)
    replay.interpolation = PoseInterpolation(
        replay.times, xyz, replay.interpolation.rotation(replay.times).as_quat(),
    )
    selection = choose_start(replay, CONFIG, "auto", 2)
    assert selection.elapsed_s == 0
    assert not selection.checks["onset_gap_s"]["passed"]


def test_pause_threshold_can_be_configured():
    selection = choose_start(motion(), CONFIG, "auto", 2, RestartThresholds(min_pause_s=2.))
    assert selection.elapsed_s == 0
    assert not selection.checks["stationary_duration_s"]["passed"]


def test_native_stationary_restart_pose_clocks_and_particle_accounting(tmp_path, monkeypatch):
    pytest.importorskip("pysplishsplash")
    from sph_drake_replay import DrakeReplay, prepare_animated_scene
    from visualize_pour import load_frames, source_poses

    origin = 31.
    source = motion(tilt=3., origin=origin)
    samples = source.interpolation
    positions, quats = samples.translation(source.times), samples.rotation(source.times).as_quat()
    recorder = TrajectoryRecorder()
    recorder.rows = [(origin + t, "source", *p, q[3], *q[:3])
                     for t, p, q in zip(source.times, positions, quats)]
    trajectory = recorder.save(tmp_path / "take.csv")
    original = trajectory.read_bytes()
    config = replace(CONFIG, source_cup=(.02, .003, .04, .004),
                     receiver_cup=(.025, .003, .04, .004), fill_fraction=.3)

    # Substitute only the map resolution in this short test to keep map
    # construction cheap. Replay still uses the native solver and VTK exporter.
    def small_maps(static, animated):
        prepare_animated_scene(static, animated)
        scene = json.loads(animated.read_text())
        for body in scene["RigidBodies"]:
            body["mapResolution"] = [12, 12, 12]
        animated.write_text(json.dumps(scene))

    monkeypatch.setattr("sph_drake_replay.prepare_animated_scene", small_maps)
    calls = []
    apply = DrakeReplay.apply

    def track_pose(replay, boundary, t, **kwargs):
        apply(replay, boundary, t, **kwargs)
        calls.append((t, kwargs["stationary"], boundary.getRigidBodyObject().getPosition().copy()))

    monkeypatch.setattr(DrakeReplay, "apply", track_pose)
    output = tmp_path / "sph"
    duration = 2.3
    report = evaluate(trajectory, output=output, duration=duration, config=config, export_fps=6.,
                      sph_start_time="auto")
    start = report["sph_start"]
    assert start["method"] == "stationary-pose restart"
    assert start["initial_zero_fluid_velocity_assumption_used"]
    dwell = start["settling_duration_s"]
    assert dwell == .5
    elapsed = start["selected_elapsed_s"]
    assert start["selected_drake_time_s"] == pytest.approx(origin + elapsed)
    assert start["onset_drake_time_s"] == pytest.approx(origin + 4., abs=.05)
    assert (output / "scene/source_trajectory.csv").read_bytes() == original
    expected, orientation = DrakeReplay(trajectory).pose(elapsed)
    assert calls[0][:2] == (elapsed, True)
    assert calls[1][:2] == (elapsed, False)
    np.testing.assert_allclose(calls[0][2], expected, atol=1e-7)
    with (output / "fluid_frames.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    clocks = np.array([[float(row[k]) for k in ("time_s", "source_pose_time_s", "drake_time_s")]
                       for row in rows])
    assert np.all(np.diff(clocks[:, 0]) > 0)
    np.testing.assert_allclose(clocks[:, 2], origin + clocks[:, 1])
    np.testing.assert_allclose(clocks[clocks[:, 0] < dwell, 1], elapsed)
    assert (clocks[:, 1] == elapsed).sum() >= 3
    assert clocks[-1, 2] > start["onset_drake_time_s"]
    assert report["applied_drake_time_s"] == pytest.approx(origin + elapsed + duration, abs=.005)
    assert report["simulation_time_s"] == pytest.approx(dwell + duration, abs=.010)
    assert 0 <= report["simulation_time_s"] - clocks[-1, 0] <= 1 / 6

    frames = load_frames(output, 6.)
    first, last = source_poses(trajectory, [frames[0][1], frames[-1][1]])
    np.testing.assert_allclose(first[1], expected)
    np.testing.assert_allclose(first[0], orientation)
    assert not np.allclose(last[0], first[0])
    initial = report["initial_fill"]
    assert initial["requested_fill_fraction"] == config.fill_fraction
    assert initial["particle_spacing_m"] == .0035
    assert abs(initial["target_fluid_mass_kg"] - initial["estimated_initialized_mass_kg"]) <= (
        initial["nominal_particle_mass_kg"] / 2
    )
    assert report["total_particles"] == initial["initialized_particle_count"]
    assert all(frame[2].shape[1] == report["total_particles"] for frame in frames)
    assert report["total_particles"] == report["in_source"] + report["in_receiver"] + report["outside_both"]
