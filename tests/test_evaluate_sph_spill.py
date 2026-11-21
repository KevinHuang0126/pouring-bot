import unittest
from collections import Counter
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np

from evaluate_sph_spill import classify, inside_cup
from prepare_sph_pouring import (
    CUP_SEGMENTS,
    RECEIVER_CUP,
    SOURCE_CUP,
    cup_mesh,
)


class SpillClassificationTest(unittest.TestCase):
    def test_faceted_cavity_excludes_solid_and_points_on_rim(self):
        radius = SOURCE_CUP[0]
        angle = np.pi / CUP_SEGMENTS
        side = np.array([np.cos(angle), np.sin(angle)]) * radius
        inside = inside_cup(np.array([
            [0, 0, 0.05],
            [0, 0, 0.002],
            [0, 0, 0.110],
            [*side, 0.05],
        ]), SOURCE_CUP)
        np.testing.assert_array_equal(inside, [True, False, False, False])

    def test_source_coordinates_follow_the_applied_pose(self):
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        origin = np.array([0, 0, 0.16])
        counts = classify(np.array([
            origin + rotation @ [0, 0, 0.08],
            [0.110, 0, 0.05],
            [0.110, 0, -0.01],
            [0.110, 0, 0.20],
        ]), origin, rotation)
        self.assertEqual(counts["in_source"], 1)
        self.assertEqual(counts["in_receiver"], 1)
        self.assertEqual(counts["outside_both"], 2)
        self.assertEqual(counts["below_world_z_zero"], 1)

    def test_generated_meshes_are_closed_and_consistently_wound(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cup.obj"
            for cup in (SOURCE_CUP, RECEIVER_CUP):
                with self.subTest(cup=cup):
                    cup_mesh(path, *cup)
                    rows = [line.split() for line in path.read_text().splitlines()]
                    vertices = np.array([r[1:] for r in rows if r[0] == "v"], dtype=float)
                    faces = np.array([r[1:] for r in rows if r[0] == "f"], dtype=int) - 1
                    triangles = vertices[faces]
                    normals = np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0])
                    self.assertTrue(np.all(np.linalg.norm(normals, axis=1) > 0))
                    self.assertGreater(np.sum(triangles[:, 0] * normals), 0)
                    edges = Counter(
                        (start, end)
                        for face in faces
                        for start, end in zip(face, np.roll(face, -1))
                    )
                    self.assertTrue(all(n == edges[end, start] == 1
                                        for (start, end), n in edges.items()))


class EvaluatorInterfaceTest(unittest.TestCase):
    def test_recorder_report_uses_configured_scene_and_isolated_outputs(self):
        import json
        import sys
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from drake_trajectory_export import TrajectoryRecorder
        from evaluate_sph_spill import PouringConfig, evaluate

        origin = np.array([0.0, 0.0, 0.16])
        recorder = TrajectoryRecorder()
        recorder.rows = [
            [time, "source", *origin, 1, 0, 0, 0] for time in (0.0, 0.1)
        ]
        config = PouringConfig(
            source_cup=(0.036, 0.003, 0.112, 0.005),
            receiver_cup=(0.061, 0.004, 0.122, 0.006),
            receiver_origin=(0.2, 0, 0),
            fill_end=(0.02, 0.02, 0.06),
        )
        replay = SimpleNamespace(times=[0, 0.1], pose=lambda _: (origin, np.eye(3)))
        binding = SimpleNamespace(
            DrakeReplay=Mock(return_value=replay),
            prepare_animated_scene=Mock(),
            replay_scene=Mock(return_value=(0.1, 0.099, np.array([
                [0, 0, 0.175], [0.2, 0, 0.005], [0.2, 0, 0.05], [0.3, 0, 0.2],
            ]))),
        )

        with TemporaryDirectory() as tmp, patch.dict(sys.modules, {"sph_drake_replay": binding}):
            root = Path(tmp)
            cache = root / "cache"
            for name in ("first", "second"):
                output = root / name
                result = evaluate(recorder, output=output, scene_dir=cache, config=config)
                self.assertEqual(result, json.loads((output / "spill_report.json").read_text()))
                self.assertIs(result["spill_measurement_validated"], False)
                self.assertEqual(result["classification"], "final_particle_centers")
                self.assertEqual(result["collected_fraction"], 0.25)
                self.assertEqual(result["remaining_fraction"], 0.25)
                self.assertEqual(result["outside_fraction"], 0.50)
                scene = output / "scene"
                self.assertEqual((scene / "Cache").resolve(), cache.resolve() / "Cache")
                payload = json.loads((scene / "scene_static.json").read_text())
                self.assertEqual(payload["RigidBodies"][1]["translation"], [0.2, 0, 0])
                np.testing.assert_allclose(payload["FluidBlocks"][0]["end"], [0.02, 0.02, 0.22])
            self.assertEqual(binding.replay_scene.call_count, 2)
            with self.assertRaisesRegex(ValueError, "new or empty"):
                evaluate(recorder, root / "first", scene_dir=cache, config=config)
            self.assertEqual(binding.replay_scene.call_count, 2)
