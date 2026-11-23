"""Conservative selection of a replay segment from measured source poses."""

import numpy as np

from prepare_sph_pouring import initial_profile_fill


def choose_start(replay, config, requested, prepour_seconds):
    """Return elapsed CSV time, measured onset (if found), and a fallback reason."""
    prepour_seconds = float(prepour_seconds)
    if not np.isfinite(prepour_seconds) or prepour_seconds < 0:
        raise ValueError("Pre-pour seconds must be finite and nonnegative")
    auto = requested == "auto"
    if not auto:
        stamp = float(requested)
        if stamp == 0:
            return 0., None, None
        start = stamp - replay.input_time_origin_s
        if not np.isfinite(start) or not 0 <= start < replay.times[-1]:
            raise ValueError("SPH start must be a timestamp within the original Drake CSV")
        if start == 0:
            return 0., None, None

    def reject(reason):
        if not auto:
            raise ValueError(f"Unsafe SPH start: {reason}. Use --sph-start-time 0 for full replay.")
        return 0., onset, reason

    onset = None
    if config.fill_fraction is None:
        return reject("No saved profile fill fraction")
    if replay.hold_source:
        return reject("The source is held fixed")

    motion = replay.interpolation
    last = replay.times[-1]
    # A uniform analysis grid keeps detection independent of control logging rate.
    times = np.linspace(0., last, max(2, int(np.ceil(last / .025)) + 1))
    _, matrices = motion.pose(times)
    tilt = np.arccos(np.clip(matrices[:, 2, 2], -1., 1.))
    if auto:
        # Find the first sustained rise followed by a pouring-scale tilt.
        # Short rotations during pickup and transport fail the rise/lookahead tests.
        step = times[1]
        sustained = max(1, round(.65 / step))
        lookahead = max(1, round(5. / step))
        rate = np.gradient(tilt, times)
        rising = rate >= np.deg2rad(3.)
        for index in np.flatnonzero(rising & ~np.r_[False, rising[:-1]]):
            if (index + sustained < len(times)
                    and rising[index:index+sustained].all()
                    and tilt[index] < np.deg2rad(35.)
                    and np.max(tilt[index:index+lookahead]) >= max(
                        np.deg2rad(45.), tilt[index] + np.deg2rad(25.))):
                onset = float(times[index])
                break
        if onset is None:
            return reject("No sustained pouring tilt was detected")
        preferred = onset - prepour_seconds
        if preferred < .5:
            return reject("Insufficient lead-in before pouring")
        choices = times[(times <= preferred) & (times >= max(.5, preferred - 3.))]
    else:
        choices = np.array([start])

    if not len(choices):
        return reject("No usable pre-pour interval")

    # Bound slosh from recent motion using point speed and changes in velocity.
    # A short window averages sample-to-sample tracking jitter. Brief impulses
    # still enter the separate kinetic-headroom check through the peak speed.
    profile = config.source_cavity()
    _, fill = initial_profile_fill(profile, config.fill_fraction, config.particle_radius)
    surface = fill["target_surface_height_m"]
    rim, rim_radius = profile.inner[-1]
    radius = max(r for _, r in profile.inner)
    lever = np.hypot(rim, radius)
    before = np.maximum(0., times - .2)
    after = np.minimum(last, times + .2)

    def rates(t):
        rotation = motion.rotation(t)
        return motion.translation(t, 1), rotation.apply(motion.rotation(t, 1))

    v, omega = rates(times)
    va, wa = rates(before)
    vb, wb = rates(after)
    speed = np.linalg.norm(v, axis=1) + lever * np.linalg.norm(omega, axis=1)
    accel = ((np.linalg.norm(vb - va, axis=1) + lever * np.linalg.norm(wb - wa, axis=1))
             / (after - before) + lever * np.linalg.norm(omega, axis=1)**2)
    effective_tilt = tilt + np.arctan2(accel, 9.81)
    headroom = (rim - surface
                - (radius + rim_radius) * np.tan(np.minimum(effective_tilt, np.pi / 2))
                - speed**2 / (2 * 9.81))
    has_safe_history = False
    for candidate in reversed(choices):
        prior = times <= candidate + times[1]
        recent = prior & (times >= candidate - .5)
        if np.any(headroom[prior] < 2 * config.particle_radius):
            continue
        has_safe_history = True
        if np.max(speed[recent]) <= .045 and np.max(accel[recent]) <= .45:
            return float(candidate), onset, None
    return reject("No sufficiently quiet start interval" if has_safe_history
                  else "Earlier motion could have brought liquid to the rim")
