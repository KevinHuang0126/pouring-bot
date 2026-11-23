import numpy as np
import pytest

from fluid_model import create_fluid_state
from prepare_sph_pouring import initial_profile_fill, write_particle_samples
from run_ml_sph_pour import hollow_profile


PROFILES = [
    [(0, .026), (1, .026)],
    [(0, .022), (.65, .022), (.82, .028), (1, .028)],
    [(0, .026), (.20, .026), (.35, .020), (.50, .020), (.72, .035), (1, .035)],
    [(0, .027), (.16, .027), (.38, .020), (.60, .020), (.84, .028), (1, .028)],
]


@pytest.mark.parametrize("knots", PROFILES, ids=["cylinder", "goblet", "waisted", "hourglass"])
def test_profile_capacity_and_quantized_fill(knots):
    profile = hollow_profile(knots, .15, .003, .004)
    angles = 2 * np.pi / profile.segments
    lo, hi = profile.inner[0][0], profile.inner[-1][0]
    z = np.linspace(lo, hi, 10001)
    areas = profile.segments * np.sin(angles) / 2 * profile.radius_at(z)**2
    assert profile.capacity_m3() == pytest.approx(np.trapezoid(areas, z), rel=1e-6)
    spacing = .003
    previous_count = 0
    for fraction in (.25, .50, .75):
        points, metrics = initial_profile_fill(profile, fraction, spacing / 2)
        assert len(points) > previous_count
        assert len(points) == metrics["initialized_particle_count"]
        assert profile.contains(points).all()
        assert len(np.unique(points, axis=0)) == len(points)
        np.testing.assert_allclose(np.diff(np.unique(points[:, 2])), spacing)
        surface = metrics["target_surface_height_m"]
        assert profile.capacity_m3(surface) == pytest.approx(profile.capacity_m3() * fraction)
        if knots is PROFILES[0]:
            assert surface == pytest.approx(lo + fraction * (hi - lo))
        assert abs(metrics["volume_error_ml"]) <= .5 * spacing**3 * 1e6
        planner = create_fluid_state("generated", fraction, capacity_ml=metrics["usable_capacity_ml"])
        assert planner.initial_volume_ml == pytest.approx(metrics["target_volume_ml"])
        assert metrics["estimated_realized_fill_fraction"] == pytest.approx(fraction, abs=.0001)
        previous_count = len(points)


def test_native_partio_round_trip(tmp_path):
    sph = pytest.importorskip("pysplishsplash")
    points = np.array([[.003, .006, .009], [.009, .006, .003], [-.009, .004, .021]])
    path = tmp_path / "initial_fluid.geo"
    write_particle_samples(path, points)
    positions = sph.Utilities.VecVector3r()
    velocities = sph.Utilities.VecVector3r()
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("x", .2).as_matrix()
    origin = np.array([.5, -.07, .3])
    assert sph.Utilities.PartioReaderWriter.readParticle(
        str(path), origin, rotation, 1., positions, velocities,
    )
    np.testing.assert_allclose(np.asarray(positions), points @ rotation.T + origin, atol=1e-7)
    np.testing.assert_allclose(np.asarray(velocities), np.zeros_like(points))


def test_capacity_that_does_not_fit_at_native_spacing_is_rejected():
    narrow = hollow_profile([(0., .0169), (1., .0169)], .12, .003, .004)
    with pytest.raises(ValueError, match="Reduce the fill fraction"):
        initial_profile_fill(narrow, 1., .0015)


@pytest.mark.parametrize("knots", PROFILES, ids=["cylinder", "goblet", "waisted", "hourglass"])
def test_coarser_spacing_conserves_nominal_fill(knots):
    profile = hollow_profile(knots, .15, .003, .004)
    _, original = initial_profile_fill(profile, .25, .0015)
    points, coarse = initial_profile_fill(profile, .25, .00175)
    assert profile.contains(points).all()
    assert coarse["target_surface_height_m"] == original["target_surface_height_m"]
    assert coarse["target_volume_ml"] == original["target_volume_ml"]
    assert coarse["initialized_particle_count"] / original["initialized_particle_count"] == pytest.approx(
        (.003/.0035)**3, abs=.002,
    )
    assert abs(coarse["volume_error_ml"]) <= .5 * coarse["nominal_particle_volume_ml"]


def test_spacing_fallback_preserves_requested_fill():
    from run_ml_sph_pour import select_particle_spacing

    profile = hollow_profile(PROFILES[-1], .15, .003, .004)
    spacing, fallback = select_particle_spacing(profile, .75, None)
    assert spacing == .003 and "Cannot seed" in fallback
    with pytest.raises(ValueError, match="Cannot seed"):
        select_particle_spacing(profile, .75, .0035)
