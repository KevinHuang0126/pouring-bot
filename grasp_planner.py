
"""Inverse kinematics for the pouring robot.

Uses the measured Schunk WSG finger midpoint and keeps all
non-iiwa degrees of freedom fixed during IK planning.
"""

import numpy as np

from pydrake.all import (
    InverseKinematics,
    RotationMatrix,
    Solve,
)

# Measured midpoint between the Schunk finger-body origins.
GRASP_POINT_G = np.array([0.0, 0.0771, 0.0])

POSITION_TOLERANCE = 0.005


def _iiwa_indices(plant, iiwa):
    """Return global position indices for the iiwa joints."""
    indices = []

    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_joint(joint_index)
        start = joint.position_start()

        indices.extend(
            range(start, start + joint.num_positions())
        )

    return np.array(indices, dtype=int)


def _solve_ik(
    plant,
    plant_context,
    target_position,
    target_rotation,
    orientation_tolerance_deg=15,
    seed_positions=None,
    attempts=10,
    perturbation=0.15,
):
    """
    Solve end-effector IK using multiple initial guesses.

    Only iiwa joint positions may change. The WSG fingers,
    vessel, and other plant positions remain fixed during IK.

    The live simulation context is never modified.
    """

    iiwa = plant.GetModelInstanceByName("iiwa")
    wsg = plant.GetModelInstanceByName("wsg")

    gripper_frame = plant.GetFrameByName("body", wsg)
    world_frame = plant.world_frame()

    q_reference = plant.GetPositions(
        plant_context
    ).copy()

    iiwa_indices = _iiwa_indices(plant, iiwa)

    if seed_positions is None:
        q_seed = q_reference.copy()
    else:
        q_seed = np.asarray(
            seed_positions, dtype=float
        ).copy()

    if len(q_seed) != plant.num_positions():
        raise ValueError(
            "IK seed must contain all plant positions."
        )

    ik = InverseKinematics(plant)
    q = ik.q()
    prog = ik.prog()

    # Freeze the non-robot positions, including the vessel.
    lower = q_reference.copy()
    upper = q_reference.copy()

    lower[iiwa_indices] = -np.inf
    upper[iiwa_indices] = np.inf

    prog.AddBoundingBoxConstraint(
        lower, upper, q
    )

    target_position = np.asarray(
        target_position, dtype=float
    )

    ik.AddPositionConstraint(
        frameB=gripper_frame,
        p_BQ=GRASP_POINT_G,
        frameA=world_frame,
        p_AQ_lower=(
            target_position - POSITION_TOLERANCE
        ),
        p_AQ_upper=(
            target_position + POSITION_TOLERANCE
        ),
    )

    ik.AddOrientationConstraint(
        frameAbar=world_frame,
        R_AbarA=target_rotation,
        frameBbar=gripper_frame,
        R_BbarB=RotationMatrix(),
        theta_bound=np.deg2rad(
            orientation_tolerance_deg
        ),
    )

    # Prefer a solution near the provided seed.
    prog.AddQuadraticErrorCost(
        np.eye(plant.num_positions()),
        q_seed,
        q,
    )

    joint_lower = plant.GetPositionLowerLimits()
    joint_upper = plant.GetPositionUpperLimits()

    rng = np.random.default_rng(42)

    last_result = None

    for attempt in range(attempts):
        guess = q_seed.copy()

        if attempt > 0:
            guess[iiwa_indices] += rng.normal(
                0.0,
                perturbation,
                len(iiwa_indices),
            )

            guess[iiwa_indices] = np.clip(
                guess[iiwa_indices],
                joint_lower[iiwa_indices] + 1e-4,
                joint_upper[iiwa_indices] - 1e-4,
            )

        prog.SetInitialGuess(q, guess)

        result = Solve(prog)
        last_result = result

        if result.is_success():
            solution = result.GetSolution(q)

            return (
                solution[iiwa_indices].copy(),
                solution.copy(),
            )

    print(
        f"IK failed after {attempts} attempts."
    )

    if last_result is not None:
        print(
            "Solver:",
            last_result.get_solver_id().name(),
        )
        print(
            "Status:",
            last_result.get_solution_result(),
        )

    print(
        "Target position:",
        np.round(target_position, 4),
    )

    print(
        "Current iiwa joints:",
        np.round(
            q_reference[iiwa_indices], 3
        ),
    )

    print(
        "Joint lower limits:",
        np.round(
            joint_lower[iiwa_indices], 3
        ),
    )

    print(
        "Joint upper limits:",
        np.round(
            joint_upper[iiwa_indices], 3
        ),
    )

    return None, None


def solve_pregrasp_ik(
    plant,
    plant_context,
    target_position,
):
    """Find a side-on grasp pose with a near-upright WSG."""

    q_iiwa, _ = _solve_ik(
        plant=plant,
        plant_context=plant_context,
        target_position=target_position,
        target_rotation=RotationMatrix(),
        orientation_tolerance_deg=15,
        attempts=15,
        perturbation=0.30,
    )

    if q_iiwa is None:
        print("Orientation-aware grasp IK failed.")
        return None

    print("Orientation-aware IK succeeded!")
    print(
        "Joint angles:",
        np.round(q_iiwa, 3),
    )

    return q_iiwa


def solve_lift_ik(
    plant,
    plant_context,
    target_position,
    R_target,
    seed_positions=None,
):
    """
    Find the next small lifting motion while preserving
    the gripper's current orientation.
    """

    q_iiwa, q_full = _solve_ik(
        plant=plant,
        plant_context=plant_context,
        target_position=target_position,
        target_rotation=R_target,
        orientation_tolerance_deg=15,
        seed_positions=seed_positions,
        attempts=12,
        perturbation=0.10,
    )

    return q_iiwa, q_full
