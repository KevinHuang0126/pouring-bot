import numpy as np
from scipy.spatial.transform import Rotation

from drake_trajectory_export import PoseInterpolation


def test_preserves_samples_and_reports_continuous_world_velocities():
    times = [0., .1, .35, .45]
    xyz = np.array([[0, 0, 0], [0, 0, 0], [.04, 0, .03], [.08, 0, .03]])
    rotation = Rotation.from_euler("xyz", [[0, 0, 0], [0, 0, 0], [10, 20, 0], [20, 23, 8]],
                                  degrees=True)
    motion = PoseInterpolation(times, xyz, rotation.as_quat())
    positions, matrices = motion.pose(times)
    np.testing.assert_allclose(positions, xyz, atol=1e-12)
    np.testing.assert_allclose(matrices, rotation.as_matrix(), atol=1e-12)
    for t in times[1:-1]:
        h = 1e-6
        before, r_before = motion.pose(t - h)
        after, r_after = motion.pose(t + h)
        v, omega = motion.velocity(t)
        position, quaternion, sampled_v, sampled_omega = motion.sample(t)
        expected_position, expected_rotation = motion.pose(t)
        np.testing.assert_allclose(position, expected_position)
        np.testing.assert_allclose(Rotation.from_quat(quaternion).as_matrix(), expected_rotation)
        np.testing.assert_allclose(sampled_v, v)
        np.testing.assert_allclose(sampled_omega, omega)
        omega_fd = Rotation.from_matrix(r_after @ r_before.T).as_rotvec() / (2*h)
        np.testing.assert_allclose(v, (after - before) / (2*h), atol=1e-5)
        np.testing.assert_allclose(omega, omega_fd, atol=1e-5)
        np.testing.assert_allclose(motion.velocity(t - h), motion.velocity(t + h), atol=1e-4)
