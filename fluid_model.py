
"""Reduced-order liquid flow and volume tracking.

Tracks:
- Initial liquid volume
- Remaining liquid volume
- Instantaneous flow rate
- Collected volume
- Spilled volume

All stored volumes are in milliliters.
Flow rates are in milliliters per second.

This is a mass-conserving pouring surrogate,
not a full fluid dynamics simulation.
"""

from dataclasses import dataclass

import numpy as np


# ==================================================
# EFFECTIVE VESSEL CAPACITIES
# ==================================================

# Approximate usable internal volumes.
#
# These are modeling assumptions based on the
# simplified vessel geometries, not measured
# capacities from real laboratory glassware.

VESSEL_CAPACITIES_ML = {
    "test_tube": 35.0,
    "erlenmeyer": 350.0,
    "boiling_flask": 350.0,
}


# ==================================================
# FLOW MODEL PARAMETERS
# ==================================================

# Fraction of the nominal mouth area that
# effectively participates in the liquid jet.
#
# A small effective area accounts approximately
# for partial opening occupancy, air entering
# the vessel, and other unmodeled effects.
#
# These are initial tuning parameters.

EFFECTIVE_AREA_FRACTIONS = {
    "test_tube": 0.05,
    "erlenmeyer": 0.05,
    "boiling_flask": 0.05,
}

ML_PER_CUBIC_METER = 1e6

EMPTY_TOLERANCE_ML = 1e-6


# ==================================================
# FLUID STATE
# ==================================================

@dataclass
class FluidState:

    vessel_name: str

    capacity_ml: float

    initial_volume_ml: float

    remaining_volume_ml: float

    collected_volume_ml: float = 0.0

    spilled_volume_ml: float = 0.0

    flow_rate_ml_s: float = 0.0

    peak_flow_rate_ml_s: float = 0.0

    elapsed_time_s: float = 0.0

    flowing_time_s: float = 0.0


    @property
    def fill_fraction(self):
        """Current fill fraction between zero and one."""

        return float(
            np.clip(
                self.remaining_volume_ml
                / self.capacity_ml,
                0.0,
                1.0,
            )
        )


    @property
    def is_empty(self):
        """Whether the source is effectively empty."""

        return (
            self.remaining_volume_ml
            <= EMPTY_TOLERANCE_ML
        )


    @property
    def transferred_volume_ml(self):
        """Total volume that has exited the source."""

        return (
            self.initial_volume_ml
            - self.remaining_volume_ml
        )


    @property
    def collection_efficiency(self):
        """Fraction of released liquid collected."""

        transferred = (
            self.transferred_volume_ml
        )

        if transferred <= EMPTY_TOLERANCE_ML:

            return 0.0

        return (
            self.collected_volume_ml
            / transferred
        )


    @property
    def collection_fraction(self):
        """Fraction of initial liquid collected."""

        return (
            self.collected_volume_ml
            / self.initial_volume_ml
        )


    @property
    def mean_flow_rate_ml_s(self):
        """Average flow rate while liquid flowed."""

        if self.flowing_time_s <= 0.0:

            return 0.0

        return (
            self.transferred_volume_ml
            / self.flowing_time_s
        )


    @property
    def conservation_error_ml(self):
        """Check conservation of liquid volume."""

        accounted_volume = (
            self.remaining_volume_ml
            + self.collected_volume_ml
            + self.spilled_volume_ml
        )

        return (
            self.initial_volume_ml
            - accounted_volume
        )


    def advance(
        self,
        dt,
        requested_flow_ml_s,
        stream_hits_receiver,
    ):
        """
        Advance the fluid model by dt seconds.

        Parameters
        ----------
        dt:
            Simulation timestep, seconds.

        requested_flow_ml_s:
            Predicted volumetric flow rate.

        stream_hits_receiver:
            True when the predicted jet enters
            the receiving container.

        Returns
        -------
        released_volume_ml:
            Volume released during this timestep.
        """

        if dt <= 0.0:

            raise ValueError(
                "dt must be positive."
            )

        if not np.isfinite(
            requested_flow_ml_s
        ):

            raise ValueError(
                "Flow rate must be finite."
            )

        self.elapsed_time_s += dt

        requested_flow_ml_s = max(
            0.0,
            float(requested_flow_ml_s),
        )

        # No liquid can leave an empty vessel.
        if (
            self.is_empty
            or requested_flow_ml_s <= 0.0
        ):

            self.flow_rate_ml_s = 0.0

            return 0.0

        # Predicted volume leaving the vessel.
        requested_volume_ml = (
            requested_flow_ml_s * dt
        )

        # Prevent negative remaining volume.
        released_volume_ml = min(
            requested_volume_ml,
            self.remaining_volume_ml,
        )

        self.remaining_volume_ml -= (
            released_volume_ml
        )

        # Account for released liquid.
        if stream_hits_receiver:

            self.collected_volume_ml += (
                released_volume_ml
            )

        else:

            self.spilled_volume_ml += (
                released_volume_ml
            )

        # Actual time-averaged flow during
        # the current simulation timestep.
        self.flow_rate_ml_s = (
            released_volume_ml / dt
        )

        # Track the peak instantaneous flow rate
        # while liquid is actively leaving the vessel.
        self.peak_flow_rate_ml_s = max(
            self.peak_flow_rate_ml_s,
            requested_flow_ml_s,
        )

        # Account for a partially active last
        # timestep when the vessel empties.
        self.flowing_time_s += (
            released_volume_ml
            / requested_flow_ml_s
        )

        return released_volume_ml


# ==================================================
# INITIALIZE FLUID STATE
# ==================================================

def create_fluid_state(
    vessel_name,
    initial_fill_fraction=0.75,
):
    """Create a new fluid state for one pouring run."""

    if vessel_name not in VESSEL_CAPACITIES_ML:

        raise ValueError(
            f"Unknown vessel: {vessel_name}"
        )

    if not (
        0.0 < initial_fill_fraction <= 1.0
    ):

        raise ValueError(
            "Initial fill fraction must be "
            "greater than zero and at most one."
        )

    capacity_ml = (
        VESSEL_CAPACITIES_ML[vessel_name]
    )

    initial_volume_ml = (
        initial_fill_fraction
        * capacity_ml
    )

    return FluidState(
        vessel_name=vessel_name,
        capacity_ml=capacity_ml,
        initial_volume_ml=initial_volume_ml,
        remaining_volume_ml=initial_volume_ml,
    )


# ==================================================
# FLOW RATE MODEL
# ==================================================

def compute_flow_rate_ml_s(
    vessel_name,
    mouth_radius,
    exit_speed_m_s,
    flow_active,
):
    """
    Estimate volumetric flow rate.

        Q = A_eff * v_exit

    The exit speed comes from the existing
    fill-dependent ballistic pouring model.

    Returns flow rate in mL/s.
    """

    if not flow_active:

        return 0.0

    if exit_speed_m_s <= 0.0:

        return 0.0

    nominal_area_m2 = (
        np.pi * mouth_radius**2
    )

    effective_area_m2 = (
        EFFECTIVE_AREA_FRACTIONS[vessel_name]
        * nominal_area_m2
    )

    flow_rate_m3_s = (
        effective_area_m2
        * exit_speed_m_s
    )

    flow_rate_ml_s = (
        flow_rate_m3_s
        * ML_PER_CUBIC_METER
    )

    return float(flow_rate_ml_s)


# ==================================================
# RESULTS
# ==================================================

def print_fluid_summary(fluid):
    """Print liquid transfer performance."""

    print("\n" + "=" * 55)
    print("FLUID TRANSFER SUMMARY")
    print("=" * 55)

    print(
        "Vessel:",
        fluid.vessel_name,
    )

    print(
        f"Vessel capacity: "
        f"{fluid.capacity_ml:.2f} mL"
    )

    print(
        f"Initial volume: "
        f"{fluid.initial_volume_ml:.2f} mL"
    )

    print(
        f"Remaining volume: "
        f"{fluid.remaining_volume_ml:.2f} mL"
    )

    print(
        f"Remaining fill fraction: "
        f"{fluid.fill_fraction:.1%}"
    )

    print(
        f"Total released volume: "
        f"{fluid.transferred_volume_ml:.2f} mL"
    )

    print(
        f"Collected volume: "
        f"{fluid.collected_volume_ml:.2f} mL"
    )

    print(
        f"Spilled volume: "
        f"{fluid.spilled_volume_ml:.2f} mL"
    )

    print(
        f"Collection efficiency: "
        f"{fluid.collection_efficiency:.1%}"
    )

    print(
        f"Initial volume collected: "
        f"{fluid.collection_fraction:.1%}"
    )

    print(
        f"Mean flow rate while flowing: "
        f"{fluid.mean_flow_rate_ml_s:.2f} mL/s"
    )

    print(
        f"Peak flow rate: "
        f"{fluid.peak_flow_rate_ml_s:.2f} mL/s"
    )

    print(
        f"Flowing time: "
        f"{fluid.flowing_time_s:.2f} s"
    )

    print(
        f"Modeled duration: "
        f"{fluid.elapsed_time_s:.2f} s"
    )

    print(
        f"Volume conservation error: "
        f"{fluid.conservation_error_ml:.8f} mL"
    )

    print("=" * 55)


# ==================================================
# INDEPENDENT FLUID MODEL TEST
# ==================================================

def test_fluid_model():

    fluid = create_fluid_state(
        "test_tube",
        initial_fill_fraction=0.75,
    )

    # Release 10 mL into the receiver.
    fluid.advance(
        dt=1.0,
        requested_flow_ml_s=10.0,
        stream_hits_receiver=True,
    )

    # Release 5 mL outside the receiver.
    fluid.advance(
        dt=1.0,
        requested_flow_ml_s=5.0,
        stream_hits_receiver=False,
    )

    assert np.isclose(
        fluid.remaining_volume_ml,
        11.25,
    )

    assert np.isclose(
        fluid.collected_volume_ml,
        10.0,
    )

    assert np.isclose(
        fluid.spilled_volume_ml,
        5.0,
    )

    assert np.isclose(
        fluid.conservation_error_ml,
        0.0,
    )

    # Request more fluid than remains.
    fluid.advance(
        dt=1.0,
        requested_flow_ml_s=100.0,
        stream_hits_receiver=True,
    )

    assert fluid.is_empty

    assert np.isclose(
        fluid.collected_volume_ml,
        21.25,
    )

    assert np.isclose(
        fluid.conservation_error_ml,
        0.0,
    )

    # An empty vessel must release no more liquid.
    released = fluid.advance(
        dt=1.0,
        requested_flow_ml_s=100.0,
        stream_hits_receiver=True,
    )

    assert released == 0.0

    print_fluid_summary(fluid)

    print("\nAll fluid model tests passed!")


if __name__ == "__main__":

    test_fluid_model()
