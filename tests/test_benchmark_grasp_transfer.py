import csv
import json
from pathlib import Path

import pytest

import benchmark_grasp_transfer as bench
from drake_trajectory_export import TrajectoryRecorder


PROFILES = {"cylinder": [(0., .026), (1., .026)], "funnel": [(0., .024), (1., .032)]}


@pytest.fixture
def simulations(monkeypatch):
    monkeypatch.setattr(bench, "load_profiles", lambda: PROFILES)
    monkeypatch.setattr(bench, "check_retirement_binding", lambda: None)
    calls = []
    failures = {}
    interrupt_after = set()

    def run_logged(script, args, log):
        if script == "run_ml_sph_pour.py":
            # --drake-only is a flag, and --grasp-height may follow it.
            def get(name):
                return args[args.index(name) + 1]

            method = get("--grasp-mode")
            run = Path(get("--output"))
            trial_id = run.parents[2].name
            calls.append((trial_id, method, "drake", list(args)))
            config = {
                "particle_radius": float(get("--particle-spacing")) / 2,
                "fill_fraction": float(get("--fill-fraction")),
            }
            package = run / "drake"
            package.mkdir(parents=True)
            failure = failures.get((trial_id, method))
            end = {"drake_time_s": 3., "target_mouth_xyz_m": [.65, .14, .46],
                   "target_axis_world": [1., 0., 0.]}
            if failure == "no_pour":
                end = None
            bench.atomic_json(package / "station_motion.json", {
                "pickup_success": failure != "pickup", "motion_completed": failure is None,
                "pour_end": end,
            })
            bench.atomic_json(package / "cups.json", config)
            bench.atomic_json(package / "vessel_geometry.json", {
                "grasp_height_m": .10 if method == "ml" else float(get("--grasp-height")),
            })
            height = json.loads(Path(get("--profile-json")).read_text())["height_m"]
            c = 2**-.5
            rec = TrajectoryRecorder()
            rec.rows = [
                [1., "source", .5, -.05, .25, 1., 0., 0., 0.],
                [3., "source", .65 - height, .14, .46, c, 0., c, 0.],
                [5., "source", .65, .14, .25, 1., 0., 0., 0.],
            ]
            if failure == "upright":
                rec.rows[1][5:] = [1., 0., 0., 0.]
                # Even a planner that claims success must pass the measured-pose check.
                status = bench.read_json(package / "station_motion.json")
                bench.atomic_json(package / "station_motion.json", {**status, "motion_completed": True})
            rec.save(package / "source_trajectory.csv")
        else:
            def get(name):
                return args[args.index(name) + 1]

            package = Path(get("--station-export"))
            method = package.parents[2].name
            trial_id = package.parents[3].name
            calls.append((trial_id, method, "sph", list(args)))
            if failures.get((trial_id, method)) == "sph":
                raise RuntimeError("Solver error")
            assert get("--sph-start-time") == "auto"
            assert get("--end-drake-time") == 3.
            assert get("--final-hold-duration") == bench.FINAL_HOLD_S
            config = bench.read_json(package / "cups.json")
            assert get("--particle-spacing") == 2 * config["particle_radius"]
            run = Path(get("--output"))
            run.mkdir(parents=True)
            receiver = 7 if method == "ml" else 4
            retire = "--retire-escaped" in args
            retired = 1 if retire else 0
            bench.atomic_json(run / "spill_report.json", {
                "total_particles": 10, "in_receiver": receiver,
                "escape_retirement": {"enabled": retire},
                "active_particles": 10 - retired, "retired_particles": retired,
                "retired_volume_ml": retired * .04, "active_outside_both": 9 - receiver - retired,
                "in_source": 1, "outside_both": 9 - receiver, "below_world_z_zero": 1,
                "particle_spacing_m": get("--particle-spacing"),
                "initial_fill": {"requested_fill_fraction": config["fill_fraction"]},
                "applied_drake_time_s": get("--end-drake-time"),
                "requested_final_hold_duration_s": get("--final-hold-duration"),
            })
        if (trial_id, method, script) in interrupt_after:
            raise KeyboardInterrupt

    monkeypatch.setattr(bench, "run_logged", run_logged)
    return calls, failures, interrupt_after


def test_randomized_trials_are_reproducible_and_paired(tmp_path, simulations):
    calls, _, _ = simulations
    settings = bench.Settings()
    expected = list(bench.sample_trials(settings, PROFILES))
    assert expected == list(bench.sample_trials(settings, PROFILES))
    assert len({trial["seed"] for trial in expected}) == 10
    assert len({trial["height_m"] for trial in expected}) == 10
    assert len({trial["mass_kg"] for trial in expected}) == 10
    assert len({trial["source_x_m"] for trial in expected}) == 10
    assert len({trial["fill_fraction"] for trial in expected}) == 10
    bench.benchmark(tmp_path)

    assert [item[1:3] for item in calls[:4]] == [
        ("ml", "drake"), ("manual", "drake"), ("ml", "sph"), ("manual", "sph"),
    ]
    for ml, manual in zip(calls[::4], calls[1::4]):
        flags = ("--profile-json", "--wall", "--floor", "--source-x", "--source-y",
                 "--fill-fraction", "--particle-spacing")
        for flag in flags:
            assert ml[3][ml[3].index(flag) + 1] == manual[3][manual[3].index(flag) + 1]
        assert "--grasp-height" not in ml[3]
        assert manual[3][manual[3].index("--grasp-height") + 1] == .09
    with (tmp_path / "trials.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 20


def test_drake_and_sph_failures_score_zero_and_preserve_counts(tmp_path, simulations):
    calls, failures, _ = simulations
    failures.update({
        ("trial_001", "ml"): "pickup",
        ("trial_002", "manual"): "upright",
        ("trial_003", "ml"): "no_pour",
        ("trial_003", "manual"): "sph",
    })
    state = bench.benchmark(tmp_path, {"trials": 3, "fixed_grasp_height": .08})
    trials = state["trials"]
    for trial, method in ((1, "ml"), (2, "manual"), (3, "ml")):
        failure = trials[trial - 1]["methods"][method]
        assert not failure["drake_success"]
        assert failure["transfer_percent"] == 0
        assert failure["failure_reason"]
        assert not any(call[:3] == (f"trial_{trial:03d}", method, "sph") for call in calls)
    assert "missed target" in trials[1]["methods"]["manual"]["failure_reason"]
    sph_error = trials[2]["methods"]["manual"]
    assert sph_error["drake_success"] and not sph_error["sph_success"]
    assert sph_error["transfer_percent"] == 0
    success = trials[0]["methods"]["manual"]
    assert success["grasp_height_m"] == .08
    assert success["transfer_percent"] == 40.
    assert success["active_particles"] == 9
    assert success["retired_particles"] == 1
    assert success["retired_volume_ml"] == .04
    assert success["active_outside_both"] == 4
    assert (success["in_source"], success["outside_both"], success["below_world_z_zero"]) == (1, 5, 1)
    summary = bench.read_json(tmp_path / "summary.json")
    assert summary["methods"]["manual"]["transfer_percent"]["mean"] == pytest.approx(40. / 3)
    assert summary["methods"]["manual"]["drake_success_rate"] == pytest.approx(2 / 3)
    assert summary["methods"]["manual"]["sph_success_rate"] == pytest.approx(1 / 3)
    assert summary["methods"]["ml"]["transfer_percent"]["median"] == 0
    assert summary["paired_ml_minus_baseline_pp"]["mean"] == 10
    assert [row["ml_minus_baseline_pp"] for row in summary["paired_ml_minus_baseline_pp"]["by_trial"]] == [
        -40., 70., 0.,
    ]


def test_resume_recovers_completed_report_and_skips_completed_stages(tmp_path, simulations, monkeypatch):
    calls, _, interrupt_after = simulations
    interrupt_after.add(("trial_002", "ml", "evaluate_sph_spill.py"))
    with pytest.raises(KeyboardInterrupt):
        bench.benchmark(tmp_path, {"trials": 2, "seed": 42})
    saved = bench.read_json(tmp_path / "trials.json")
    assert saved["trials"][0]["methods"]["ml"]["status"] == "complete"
    assert saved["trials"][1]["methods"]["ml"]["status"] == "drake_complete"
    monkeypatch.setattr(bench, "load_profiles", lambda: pytest.fail("Resume sampled again"))
    with pytest.raises(ValueError, match="Resume options"):
        bench.benchmark(tmp_path, {"seed": 7}, resume=True)

    calls.clear()
    bench.benchmark(tmp_path, resume=True)
    assert [call[:3] for call in calls] == [("trial_002", "manual", "sph")]
    calls.clear()
    bench.benchmark(tmp_path, resume=True)
    assert not calls


def test_legacy_resume_keeps_retirement_disabled(tmp_path, simulations):
    calls, _, interrupt_after = simulations
    interrupt_after.add(("trial_001", "ml", "evaluate_sph_spill.py"))
    with pytest.raises(KeyboardInterrupt):
        bench.benchmark(tmp_path, {"trials": 1, "retire_escaped": False})
    manifest = tmp_path / "trials.json"
    state = bench.read_json(manifest)
    state["settings"].pop("retire_escaped")
    state["settings"].pop("escape_plane_z")
    bench.atomic_json(manifest, state)
    report_path = tmp_path / state["trials"][0]["methods"]["ml"]["sph_report"]
    report = bench.read_json(report_path)
    for key in (*bench.RETIREMENT_FIELDS, "escape_retirement"):
        report.pop(key)
    bench.atomic_json(report_path, report)
    original = report_path.read_bytes()
    calls.clear()
    result = bench.benchmark(tmp_path, resume=True)
    assert [call[:3] for call in calls] == [("trial_001", "manual", "sph")]
    assert "--retire-escaped" not in calls[0][3]
    assert result["trials"][0]["methods"]["ml"]["transfer_percent"] == 70.
    assert report_path.read_bytes() == original
