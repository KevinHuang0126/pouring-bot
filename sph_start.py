"""Select a stationary restart from measured source poses."""

from dataclasses import asdict, dataclass

import numpy as np

from prepare_sph_pouring import initial_profile_fill


@dataclass(frozen=True)
class RestartThresholds:
    min_pause_s: float = .75
    max_onset_gap_s: float = 1.25
    smoothing_window_s: float = .15
    max_speed_m_s: float = .016
    max_angular_speed_deg_s: float = 7.
    max_acceleration_m_s2: float = .2
    max_angular_acceleration_deg_s2: float = 80.
    max_tilt_deg: float = 10.

    def __post_init__(self):
        if any(not np.isfinite(value) or value <= 0 for value in asdict(self).values()):
            raise ValueError("Stationary restart thresholds must be finite and positive")
        if self.max_tilt_deg >= 90:
            raise ValueError("Stationary restart tilt threshold must be below 90 degrees")


@dataclass(frozen=True)
class StartSelection:
    elapsed_s: float = 0.
    onset_s: float | None = None
    fallback_reason: str | None = None
    interval_s: tuple[float, float] | None = None
    checks: dict | None = None


def pour_onset(times, tilt):
    """Find the first sustained rise followed by a pouring-scale tilt."""
    step = times[1]
    sustained = max(1, round(.65 / step))
    lookahead = max(1, round(5. / step))
    rising = np.gradient(tilt, times) >= np.deg2rad(3.)
    for index in np.flatnonzero(rising & ~np.r_[False, rising[:-1]]):
        if (index + sustained < len(times)
                and rising[index:index+sustained].all()
                and tilt[index] < np.deg2rad(35.)
                and np.max(tilt[index:index+lookahead]) >= max(
                    np.deg2rad(45.), tilt[index] + np.deg2rad(25.))):
            return float(times[index])
    return None


def choose_start(replay, config, requested, prepour_seconds, thresholds=RestartThresholds()):
    """Select elapsed CSV time and report measurements supporting the decision.

    A quiet vessel provides time for slosh to decay. This is a proxy for fluid
    state: the restart seeds the saved fill at rest and cannot recover prior loss.
    """
    prepour_seconds = float(prepour_seconds)
    if not np.isfinite(prepour_seconds) or prepour_seconds < 0:
        raise ValueError("Pre-pour seconds must be finite and nonnegative")
    auto = requested == "auto"
    last = replay.times[-1]
    if not auto:
        stamp = float(requested)
        if stamp == 0:
            return StartSelection()
        start = stamp - replay.input_time_origin_s
        if not np.isfinite(start) or not 0 <= start < last:
            raise ValueError("SPH start must be a timestamp within the original Drake CSV")
        if start == 0:
            return StartSelection()

    onset = None

    def reject(reason, interval=None, checks=None):
        if not auto:
            raise ValueError(f"Unsafe SPH start: {reason}. Use --sph-start-time 0 for full replay.")
        return StartSelection(onset_s=onset, fallback_reason=reason, interval_s=interval,
                              checks=checks)

    motion = replay.interpolation
    # A uniform analysis grid keeps derivative estimates independent of logging rate.
    times = np.linspace(0., last, max(2, int(np.ceil(last / .025)) + 1))
    tilt = np.arccos(np.clip(motion.rotation(times).as_matrix()[:, 2, 2], -1., 1.))
    onset = pour_onset(times, tilt) if auto else None
    if config.fill_fraction is None:
        return reject("No saved profile fill fraction")
    if replay.hold_source:
        return reject("The source is held fixed")
    if auto and onset is None:
        return reject("No sustained pouring tilt was detected")

    # Average displacement and relative rotation over a short window, then
    # differentiate their world-frame rates to estimate recent acceleration.
    half_window = thresholds.smoothing_window_s / 2
    before = np.maximum(0., times - half_window)
    after = np.minimum(last, times + half_window)
    width = (after - before)[:, None]
    velocity = (motion.translation(after) - motion.translation(before)) / width
    angular_velocity = (motion.rotation(after) * motion.rotation(before).inv()).as_rotvec() / width
    speed = np.linalg.norm(velocity, axis=1)
    angular_speed = np.linalg.norm(angular_velocity, axis=1)
    acceleration = np.linalg.norm(np.gradient(velocity, times, axis=0), axis=1)
    angular_acceleration = np.linalg.norm(np.gradient(angular_velocity, times, axis=0), axis=1)

    motion_limits = {
        "translation_speed_m_s": (speed, thresholds.max_speed_m_s),
        "angular_speed_deg_s": (np.rad2deg(angular_speed), thresholds.max_angular_speed_deg_s),
        "translation_acceleration_m_s2": (acceleration, thresholds.max_acceleration_m_s2),
        "angular_acceleration_deg_s2": (np.rad2deg(angular_acceleration),
                                        thresholds.max_angular_acceleration_deg_s2),
    }
    quiet = np.logical_and.reduce([values <= limit for values, limit in motion_limits.values()])
    if auto:
        quiet &= times < onset
    edges = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
    runs = list(zip(edges[::2], edges[1::2] - 1))
    if auto:
        recent = [(lo, hi) for lo, hi in runs if onset - times[hi] <= thresholds.max_onset_gap_s]
        if not runs:
            return reject("No pre-pour samples met all motion limits", checks={
                "quiet_samples": {"measured": 0, "limit": 1, "comparison": ">=", "passed": False},
            })
        lo, hi = next(((lo, hi) for lo, hi in reversed(recent)
                       if times[hi] - times[lo] >= thresholds.min_pause_s), runs[-1])
    else:
        bounds = next(((lo, hi) for lo, hi in runs if times[lo] <= start <= times[hi]), None)
        if bounds is None:
            return reject("The requested timestamp is outside a quiet interval")
        lo, hi = bounds

    interval = float(times[lo]), float(times[hi])
    duration = interval[1] - interval[0]
    if auto:
        # Aim for the requested lead-in, keeping enough measured quiet time
        # after the chosen pose. If the pause starts later, use its first pose.
        preferred = min(onset - prepour_seconds, interval[1] - thresholds.min_pause_s)
        index = max(lo, int(np.searchsorted(times, preferred, side="right")) - 1)
        start = float(times[index])

    profile = config.source_cavity()
    _, fill = initial_profile_fill(profile, config.fill_fraction, config.particle_radius)
    surface = fill["target_surface_height_m"]
    rim, rim_radius = profile.inner[-1]
    radius = max(r for _, r in profile.inner)
    lever = np.hypot(rim, radius)
    point_accel = acceleration + lever * (angular_acceleration + angular_speed**2)
    effective_tilt = tilt + np.arctan2(point_accel, 9.81)
    point_speed = speed + lever * angular_speed
    headroom = np.where(effective_tilt < np.pi / 2,
                        rim - surface - (radius + rim_radius) * np.tan(effective_tilt)
                        - point_speed**2 / (2 * 9.81), -np.inf)

    def check(measured, limit, *, minimum=False):
        return {
            "measured": measured if np.isfinite(measured) else None,
            "limit": limit,
            "comparison": ">=" if minimum else "<=",
            "passed": bool(measured >= limit if minimum else measured <= limit),
        }

    section = slice(lo, hi + 1)
    checks = {
        "stationary_duration_s": check(duration, thresholds.min_pause_s, minimum=True),
        "stationary_duration_after_start_s": check(interval[1] - start, thresholds.min_pause_s,
                                                  minimum=True),
        **{name: check(float(np.max(values[section])), limit)
           for name, (values, limit) in motion_limits.items()},
        "tilt_deg": check(float(np.max(np.rad2deg(tilt[section]))), thresholds.max_tilt_deg),
        "rim_headroom_m": check(float(np.min(headroom[section])),
                                2 * config.particle_radius, minimum=True),
    }
    if auto:
        checks["onset_gap_s"] = check(onset - interval[1], thresholds.max_onset_gap_s)
    failed = [name for name, result in checks.items() if not result["passed"]]
    if failed:
        return reject("Stationary restart checks failed: " + ", ".join(failed), interval, checks)
    if start == 0:
        return reject("The stationary interval leaves no motion to skip", interval, checks)
    return StartSelection(start, onset, interval_s=interval, checks=checks)
