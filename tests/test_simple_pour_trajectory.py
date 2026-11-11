"""Pure NumPy tests; no Drake needed for the trajectory mathematics."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simple_pour_trajectory import (VESSELS, compute_pour_trajectory,
                                    target_opening_error)


def tilted_pose(angle_deg, height=0.6):
    theta = np.deg2rad(angle_deg)
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]]
    pose[:3, 3] = [0.5, 0.0, height]
    return pose


class TestPourTrajectory(unittest.TestCase):
    def test_empty_never_pours(self):
        result = compute_pour_trajectory(tilted_pose(80), VESSELS['test_tube'], 0)
        self.assertFalse(result.active)
        self.assertEqual(result.points_W.shape, (0, 3))

    def test_upright_does_not_pour(self):
        result = compute_pour_trajectory(tilted_pose(0), VESSELS['test_tube'], .95)
        self.assertFalse(result.active)

    def test_tilted_flask_has_ballistic_jet(self):
        result = compute_pour_trajectory(tilted_pose(60), VESSELS['erlenmeyer'], .8)
        self.assertTrue(result.active)
        self.assertGreater(len(result.points_W), 3)
        np.testing.assert_allclose(result.points_W[0], result.mouth_W)
        self.assertAlmostEqual(result.points_W[-1, 2], .20, places=8)
        self.assertGreater(result.points_W[-1, 0], result.mouth_W[0])
        self.assertTrue(np.all(np.diff(result.points_W[:, 2]) < 0))

    def test_fill_changes_onset(self):
        pose = tilted_pose(50)
        high = compute_pour_trajectory(pose, VESSELS['test_tube'], .9)
        low = compute_pour_trajectory(pose, VESSELS['test_tube'], .1)
        self.assertTrue(high.active)
        self.assertFalse(low.active)

    def test_moving_vessel_changes_initial_velocity(self):
        pose = tilted_pose(60)
        still = compute_pour_trajectory(pose, VESSELS['test_tube'], .8)
        moving = compute_pour_trajectory(pose, VESSELS['test_tube'], .8,
                                         linear_velocity_W=(0.25, 0, 0))
        self.assertAlmostEqual(moving.launch_velocity_W[0] -
                               still.launch_velocity_W[0], .25)

    def test_target_opening_score(self):
        result = compute_pour_trajectory(tilted_pose(60), VESSELS['erlenmeyer'], .8)
        idx = len(result.points_W) // 2
        xy = result.points_W[idx, :2]
        z = result.points_W[idx, 2]
        hit, miss = target_opening_error(result, xy, z, .04)
        self.assertTrue(hit)
        self.assertAlmostEqual(miss, 0, places=7)
        no_hit, miss = target_opening_error(result, xy + [1, 1], z, .04)
        self.assertFalse(no_hit)
        self.assertGreater(miss, 1)

    def test_reject_bad_fill(self):
        with self.assertRaises(ValueError):
            compute_pour_trajectory(tilted_pose(60), VESSELS['test_tube'], 1.1)


if __name__ == '__main__':
    unittest.main()
