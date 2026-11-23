import csv
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from evaluate_sph_spill import DEFAULT_TEMPLATE, classify
from prepare_sph_pouring import PouringConfig, prepare_scene
from run_ml_sph_pour import hollow_profile
from visualize_pour import source_poses


def test_profile_offset_keeps_silhouette_and_closes_floor():
    exact = [(0., .024), (.24, .024), (.27, .033), (.59, .018), (1., .022)]
    profile = hollow_profile(exact, .15, .003, .004)
    np.testing.assert_allclose(profile.outer, [(z * .15, r) for z, r in exact])
    assert profile.segments == 48
    assert profile.inner[0] == (.004, .021)
    assert profile.contains([[.02, 0., .005], [.02, 0., .09], [0., 0., .151]]).tolist() == [
        True, False, False,
    ]
    with pytest.raises(ValueError, match="positive radius"):
        hollow_profile(exact, .15, .018, .004)


def test_nonzero_time_and_tilted_scene_use_measured_frames(tmp_path):
    trajectory = tmp_path / "source.csv"
    source = Rotation.from_euler("y", .25)
    receiver = Rotation.from_euler("x", -.4)
    origin = np.array([.48, -.1, .32])
    with trajectory.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("time_s", "body", "x_m", "y_m", "z_m", "qw", "qx", "qy", "qz"))
        writer.writerow((17., "source", *origin, *(2 * source.as_quat(scalar_first=True))))
        writer.writerow((17.5, "source", *(origin + [.1, 0, 0]), *source.as_quat(scalar_first=True)))
    config = PouringConfig(receiver_origin=(.72, .14, .2),
                           receiver_quaternion_wxyz=receiver.as_quat(scalar_first=True))
    out = tmp_path / "scene"
    scene = json.loads(prepare_scene(trajectory, out, DEFAULT_TEMPLATE, settings=config).read_text())
    assert scene["initialFillFrame"] == "source"
    assert scene["FluidBlocks"][0]["start"] == list(config.fill_start)
    assert scene["RigidBodies"][0]["translation"] == list(origin)
    for body, rotation in zip(scene["RigidBodies"], (source, receiver)):
        np.testing.assert_allclose(body["rotationAngle"] * np.asarray(body["rotationAxis"]),
                                   rotation.as_rotvec())
    for (matrix, xyz), offset in zip(source_poses(trajectory, [0., .5]), [0., .1]):
        np.testing.assert_allclose(matrix, source.as_matrix())
        np.testing.assert_allclose(xyz, origin + [offset, 0, 0])

    receiver_inside = np.array(config.receiver_origin) + receiver.apply([0., .058, .116])
    counts = classify(np.array([origin + source.apply([0., 0., .02]), receiver_inside]),
                      origin, source.as_matrix(), config=config)
    assert (counts["in_source"], counts["in_receiver"], counts["outside_both"]) == (1, 1, 0)
