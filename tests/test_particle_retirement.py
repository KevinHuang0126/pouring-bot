import csv
import json

import numpy as np
import pytest

from drake_trajectory_export import TrajectoryRecorder
from evaluate_sph_spill import DEFAULT_TEMPLATE, classify
from prepare_sph_pouring import PouringConfig, prepare_scene


def test_retirement_keeps_original_scoring_denominator():
    points = np.array([[.110, 0., .05], [0., 0., .19], [.4, .4, .2]])
    result = classify(points, [0., 0., .16], np.eye(3), retired_particles=2, retired_below_zero=1)
    assert result["total_particles"] == 5
    assert result["active_particles"] == 3
    assert result["in_receiver"] == 1
    assert result["collected_fraction"] == .2
    assert result["outside_both"] == 3
    assert result["active_outside_both"] == 1
    assert result["below_world_z_zero"] == 1
    empty = classify(np.empty((0, 3)), [0., 0., .16], np.eye(3), retired_particles=2)
    assert empty["total_particles"] == empty["retired_particles"] == empty["outside_both"] == 2


def test_native_retirement_preserves_survivor_state_and_volume(tmp_path):
    sph = pytest.importorskip("pysplishsplash")
    if not hasattr(sph.FluidModel, "retireEscapedParticles"):
        pytest.skip("Retirement binding has not been built")
    from sph_drake_replay import DrakeReplay, EscapeRetirement, prepare_animated_scene, replay_scene

    recorder = TrajectoryRecorder()
    recorder.rows = [[t, "source", 0., 0., .3, 1., 0., 0., 0.] for t in (0., .10)]
    trajectory = recorder.save(tmp_path / "source.csv")
    replay = DrakeReplay(trajectory, hold_source=True)
    config = PouringConfig(source_cup=(.02, .003, .04, .004), receiver_cup=(.025, .003, .04, .004),
                           receiver_origin=(.2, 0., .2), fill_fraction=.2, particle_radius=.00175, max_time_step=.0001)
    scene = prepare_scene(trajectory, tmp_path / "scene", DEFAULT_TEMPLATE, settings=config, export_fps=0.)
    animated = scene.with_name("scene_animated.json")
    prepare_animated_scene(scene, animated)
    payload = json.loads(animated.read_text())
    for body in payload["RigidBodies"]:
        body["mapResolution"] = [24, 24, 24]
    animated.write_text(json.dumps(payload))

    outcome = []
    for enabled in (False, True):
        output = tmp_path / str(enabled)
        history = []
        tracked = []

        def sample(simulation, _, end):
            if end < .04:
                return
            model = simulation.getFluidModel(0)
            n = model.numActiveParticles()
            p = np.asarray(model.getFieldBuffer("position"))[:n]
            v = np.asarray(model.getFieldBuffer("velocity"))[:n]
            ids = np.asarray(model.getFieldBuffer("id"))[:n]
            if not history:
                indices = [0, 2, 4, 6]
                tracked.extend(int(ids[i]) for i in indices)
                # Isolated downward, isolated upward, and a downward particle
                # linked to a survivor just above the cutoff.
                p[indices] = [[-1, 0, -.10], [-.5, 0, -.02], [-.7, 0, -.002], [-.7, 0, .002]]
                v[indices] = [[0, 0, -.04], [0, 0, .10], [0, 0, -.04], [0, 0, .04]]
            order = np.argsort(ids)
            history.append((end, ids[order], p[order].copy()))
            if enabled and len(history) == 1:
                fields = ("position", "velocity", "density", "p / rho^2", "p_v / rho^2", "velocity difference")
                keep = ids != tracked[0]
                expected = {name: np.asarray(model.getFieldBuffer(name))[:n][keep].copy()
                            for name in fields}
                expected_ids = ids[keep].copy()
                retirement.advance(model)
                survivor_count = model.numActiveParticles()
                assert retirement.count == 1
                np.testing.assert_array_equal(ids[:survivor_count], expected_ids)
                for name in fields:
                    np.testing.assert_array_equal(np.asarray(model.getFieldBuffer(name))[:survivor_count],
                                                  expected[name])

        retirement = EscapeRetirement(requested_plane_z=0.) if enabled else None
        replay_scene(animated, replay, .065, output, headless=True, retirement=retirement, on_step=sample)
        with (output / "replay_samples.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert "active_particles" in rows[0] and "retired_particles" in rows[0]
        assert "peak_particle_speed_m_s" in rows[0]
        outcome.append((history, tracked, retirement))

    control, control_ids, _ = outcome[0]
    culled, culled_ids, retirement = outcome[1]
    assert control_ids == culled_ids
    downward, upward, linked, bridge = culled_ids
    assert downward not in culled[1][1]
    assert all(particle in culled[1][1] for particle in (upward, linked, bridge))
    assert all(particle not in culled[-1][1] for particle in (downward, upward))
    assert retirement.count == len(culled[0][1]) - len(culled[-1][1])
    assert retirement.count == len(set(culled[0][1]) - set(culled[-1][1]))
    assert retirement.below_zero == retirement.count
    assert retirement.volume_m3 == pytest.approx(retirement.count * (2 * config.particle_radius)**3)
    assert len(control) == len(culled)
    peak_position_error = 0.
    for off_step, on_step in zip(control, culled):
        assert off_step[0] == on_step[0]
        off_mask = ~np.isin(off_step[1], control_ids)
        on_mask = ~np.isin(on_step[1], culled_ids)
        np.testing.assert_array_equal(off_step[1][off_mask], on_step[1][on_mask])
        peak_position_error = max(peak_position_error, np.max(abs(off_step[2][off_mask] - on_step[2][on_mask])))
    # Native reduction and iteration order can differ after compaction.
    assert peak_position_error < .01 * (2 * config.particle_radius)

    def drain(simulation, start, _):
        if start == 0.:
            model = simulation.getFluidModel(0)
            np.asarray(model.getFieldBuffer("position"))[:, 2] = -.10
            np.asarray(model.getFieldBuffer("velocity"))[:, 2] = -1.

    from visualize_pour import load_frames

    retirement = EscapeRetirement(requested_plane_z=0.)
    payload["Configuration"]["enableVTKExport"] = True
    animated.write_text(json.dumps(payload))
    empty_output = tmp_path / "empty"
    _, _, final_positions = replay_scene(animated, replay, .002, empty_output,
                                         retirement=retirement, on_step=drain)
    assert not len(final_positions)
    assert retirement.count == len(culled[0][1])
    frames = load_frames(empty_output, payload["Configuration"]["dataExportFPS"])
    assert frames[0][2].shape == (3, 0)


def test_escape_plane_accounts_for_swept_orientation_and_kernel_clearance():
    pytest.importorskip("pysplishsplash")
    from types import SimpleNamespace
    from sph_drake_replay import EscapeRetirement

    simulation = SimpleNamespace(GRAVITATION=0, numberOfFluidModels=lambda: 1,
                                 is2DSimulation=lambda: False, getSupportRadius=lambda: .01,
                                 getVec3ValueReal=lambda _: np.array([0., 0., -9.81]))
    replay = SimpleNamespace(hold_source=False, times=np.array([0., 1., 2.]),
                             interpolation=SimpleNamespace(translation=lambda t: np.c_[t*0, t*0, .2 + t*.01]))
    receiver = SimpleNamespace(getVertices=lambda: [[0., 0., .16]])
    retirement = EscapeRetirement()
    retirement.configure(simulation, replay, 0., 2., None, receiver, .15)
    assert retirement.geometry_lower_bound_z == pytest.approx(.05)
    assert retirement.plane_z == pytest.approx(-.03)
    with pytest.raises(ValueError, match="Escape plane must be at or below"):
        EscapeRetirement(.04).configure(simulation, replay, 0., 2., None, receiver, .15)
