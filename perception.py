
"""Camera-based source vessel localization.

A fixed overhead RGB-D camera estimates the
initial world-frame x and y position of a vessel.

Assumptions:
- Vessel geometry is known.
- Vessel starts upright.
- Camera is calibrated and fixed.
- Drake segmentation labels identify vessel pixels.

No ground-truth vessel position is used by
the position estimator.
"""

import numpy as np

from pydrake.all import (
    Role,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Sphere,
)


# ==================================================
# CAMERA PARAMETERS
# ==================================================

CAMERA_NAME = "overhead"

CAMERA_X = 0.49
CAMERA_Y = -0.065
CAMERA_Z = 1.15

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Minimum number of visible vessel pixels.
MIN_VISIBLE_PIXELS = 20

# Thickness of the top surface region used
# for estimating the vessel center.
TOP_SURFACE_BAND = 0.005


# ==================================================
# CAMERA CONFIGURATION
# ==================================================

def camera_config_yaml():
    """
    Return the YAML configuration for a fixed
    overhead RGB-D camera.

    Drake cameras look along their local +z axis.

    Rotating 180 degrees about world X points
    the camera downward toward the table.
    """

    return f"""
cameras:
    overhead:
        name: {CAMERA_NAME}

        width: {CAMERA_WIDTH}
        height: {CAMERA_HEIGHT}

        rgb: true
        depth: true
        label: true

        X_PB:
            base_frame: world
            translation: [{CAMERA_X}, {CAMERA_Y}, {CAMERA_Z}]
            rotation: !Rpy {{ deg: [180, 0, 0] }}
"""


def get_camera_pose():
    """
    Return the known, calibrated camera pose.

    X_WC maps camera-frame points into
    the world frame.
    """

    return RigidTransform(
        RotationMatrix.MakeXRotation(np.pi),
        np.array([
            CAMERA_X,
            CAMERA_Y,
            CAMERA_Z,
        ]),
    )


# ==================================================
# READ CAMERA IMAGES
# ==================================================

def get_camera_data(
    station,
    station_context,
):
    """
    Read the current RGB, depth, and label images.

    Depth is returned in meters.

    Camera intrinsics are obtained from
    Drake's actual camera configuration.
    """

    rgb = (
        station.GetOutputPort(
            f"{CAMERA_NAME}.rgb_image"
        )
        .Eval(station_context)
        .data.copy()
    )

    depth = (
        station.GetOutputPort(
            f"{CAMERA_NAME}.depth_image"
        )
        .Eval(station_context)
        .data.squeeze().copy()
    )

    labels = (
        station.GetOutputPort(
            f"{CAMERA_NAME}.label_image"
        )
        .Eval(station_context)
        .data.squeeze().copy()
    )

    # Access the camera system for its
    # intrinsic calibration parameters.
    camera = station.GetSubsystemByName(
        f"rgbd_sensor_{CAMERA_NAME}"
    )

    cam_info = (
        camera.default_depth_render_camera()
        .core()
        .intrinsics()
    )

    X_WC = get_camera_pose()

    return (
        rgb,
        depth,
        labels,
        cam_info,
        X_WC,
    )


# ==================================================
# IDENTIFY THE SOURCE VESSEL'S RENDER LABELS
# ==================================================

def get_vessel_render_labels(
    station,
    plant,
    vessel_body,
):
    """
    Find the segmentation labels associated
    with the known source vessel.

    This accesses geometry metadata only.
    It does NOT query the vessel's position.

    Different visual geometries belonging to
    the same vessel can have different labels.
    """

    scene_graph = station.GetSubsystemByName(
        "scene_graph"
    )

    inspector = scene_graph.model_inspector()

    frame_id = plant.GetBodyFrameIdOrThrow(
        vessel_body.index()
    )

    geometry_ids = inspector.GetGeometries(
        frame_id,
        Role.kPerception,
    )

    vessel_labels = set()

    for geometry_id in geometry_ids:

        properties = (
            inspector.GetPerceptionProperties(
                geometry_id
            )
        )

        if properties is None:
            continue

        if properties.HasProperty(
            "label",
            "id",
        ):

            render_label = properties.GetProperty(
                "label",
                "id",
            )

            vessel_labels.add(
                int(render_label)
            )

    if not vessel_labels:

        raise RuntimeError(
            "Could not find render labels "
            "for the source vessel."
        )

    return sorted(vessel_labels)


# ==================================================
# SEGMENT THE SOURCE VESSEL
# ==================================================

def segment_vessel(
    label_image,
    depth_image,
    vessel_labels,
):
    """
    Create a Boolean mask of the source vessel.

    Exclude invalid depth measurements.
    """

    label_mask = np.isin(
        label_image,
        vessel_labels,
    )

    valid_depth = (
        np.isfinite(depth_image)
        & (depth_image > 0.0)
    )

    mask = (
        label_mask
        & valid_depth
    )

    pixel_count = int(
        np.count_nonzero(mask)
    )

    if pixel_count < MIN_VISIBLE_PIXELS:

        raise RuntimeError(
            f"Source vessel not sufficiently "
            f"visible: {pixel_count} pixels."
        )

    return mask


# ==================================================
# BACK-PROJECT DEPTH PIXELS
# ==================================================

def project_depth_to_camera(
    depth_image,
    mask,
    cam_info,
):
    """
    Convert masked depth pixels into 3D
    camera-frame points.

    Same pinhole camera projection used
    in PS4 normal_estimation.py.

    Returns an N x 3 NumPy array.
    """

    # Image coordinates:
    # v = row, u = column.
    v, u = np.nonzero(mask)

    Z = depth_image[v, u]

    fx = cam_info.focal_x()
    fy = cam_info.focal_y()

    cx = cam_info.center_x()
    cy = cam_info.center_y()

    X = (
        (u - cx)
        * Z
        / fx
    )

    Y = (
        (v - cy)
        * Z
        / fy
    )

    points_C = np.column_stack([
        X,
        Y,
        Z,
    ])

    return points_C


def transform_points_to_world(
    points_C,
    X_WC,
):
    """
    Transform camera-frame points into
    world-frame points.
    """

    R_WC = (
        X_WC.rotation().matrix()
    )

    p_WC = X_WC.translation()

    points_W = (
        points_C @ R_WC.T
        + p_WC
    )

    return points_W


# ==================================================
# ESTIMATE VESSEL CENTER
# ==================================================

def estimate_vessel_center(
    points_W,
):
    """
    Estimate the horizontal center of an
    upright rotationally symmetric vessel.

    Use points near the top of the vessel,
    rather than averaging all visible points.

    This helps reduce bias caused by the wider
    base of the Erlenmeyer and boiling flasks.
    """

    if len(points_W) == 0:

        raise RuntimeError(
            "Empty vessel point cloud."
        )

    # Highest observed surface.
    z_max = float(
        np.max(points_W[:, 2])
    )

    # Select points close to the top surface.
    top_mask = (
        points_W[:, 2]
        >= z_max - TOP_SURFACE_BAND
    )

    top_points = points_W[
        top_mask
    ]

    if len(top_points) < MIN_VISIBLE_PIXELS:

        raise RuntimeError(
            "Not enough visible top-surface "
            "points for reliable localization."
        )

    # Center of the observed top-surface
    # bounding box in world x and y.
    xy_min = np.min(
        top_points[:, :2],
        axis=0,
    )

    xy_max = np.max(
        top_points[:, :2],
        axis=0,
    )

    center_xy = (
        xy_min + xy_max
    ) / 2.0

    estimated_x = float(
        center_xy[0]
    )

    estimated_y = float(
        center_xy[1]
    )

    return (
        estimated_x,
        estimated_y,
        top_points,
    )


# ==================================================
# VISUALIZE PERCEPTION RESULTS
# ==================================================

def visualize_detection(
    meshcat,
    estimated_x,
    estimated_y,
    grasp_z,
    points_W,
):
    """
    Draw a red marker at the estimated
    grasp location in Meshcat.
    """

    marker_position = np.array([
        estimated_x,
        estimated_y,
        grasp_z,
    ])

    meshcat.SetObject(
        "perception/estimated_grasp",
        Sphere(0.008),
        Rgba(1.0, 0.1, 0.1, 1.0),
    )

    meshcat.SetTransform(
        "perception/estimated_grasp",
        RigidTransform(marker_position),
    )

    # Draw a blue marker at the estimated
    # center of the vessel's visible top.
    top_center = np.array([
        estimated_x,
        estimated_y,
        np.max(points_W[:, 2]),
    ])

    meshcat.SetObject(
        "perception/estimated_top",
        Sphere(0.005),
        Rgba(0.1, 0.5, 1.0, 1.0),
    )

    meshcat.SetTransform(
        "perception/estimated_top",
        RigidTransform(top_center),
    )


# ==================================================
# COMPLETE LOCALIZATION PIPELINE
# ==================================================

def detect_vessel_position(
    station,
    station_context,
    plant,
    vessel_body,
    meshcat=None,
    grasp_z=None,
):
    """
    Estimate the source vessel's initial x and y
    coordinates using camera observations.

    The vessel body is used only to identify
    its render labels, not to read its pose.

    Returns:
        estimated_x, estimated_y
    """

    print("\n" + "=" * 55)
    print("CAMERA-BASED LOCALIZATION")
    print("=" * 55)

    # ----------------------------------------------
    # CAMERA OBSERVATION
    # ----------------------------------------------

    (
        rgb,
        depth,
        labels,
        cam_info,
        X_WC,
    ) = get_camera_data(
        station,
        station_context,
    )

    print(
        "RGB image shape:",
        rgb.shape,
    )

    print(
        "Depth image shape:",
        depth.shape,
    )

    # ----------------------------------------------
    # VESSEL SEGMENTATION
    # ----------------------------------------------

    vessel_labels = (
        get_vessel_render_labels(
            station,
            plant,
            vessel_body,
        )
    )

    print(
        "Source vessel labels:",
        vessel_labels,
    )

    mask = segment_vessel(
        labels,
        depth,
        vessel_labels,
    )

    print(
        "Detected vessel pixels:",
        np.count_nonzero(mask),
    )

    # ----------------------------------------------
    # PIXEL TO 3D POINTS
    # ----------------------------------------------

    points_C = (
        project_depth_to_camera(
            depth,
            mask,
            cam_info,
        )
    )

    # ----------------------------------------------
    # CAMERA FRAME TO WORLD FRAME
    # ----------------------------------------------

    points_W = (
        transform_points_to_world(
            points_C,
            X_WC,
        )
    )

    # ----------------------------------------------
    # ESTIMATE CENTER
    # ----------------------------------------------

    (
        estimated_x,
        estimated_y,
        top_points,
    ) = estimate_vessel_center(
        points_W
    )

    print(
        "Top-surface points:",
        len(top_points),
    )

    print(
        f"Estimated source x: "
        f"{estimated_x:.5f} m"
    )

    print(
        f"Estimated source y: "
        f"{estimated_y:.5f} m"
    )

    print(
        "Estimated vessel top height:",
        round(
            np.max(points_W[:, 2]),
            4,
        ),
        "m",
    )

    # ----------------------------------------------
    # MESHCAT DEBUG VISUALIZATION
    # ----------------------------------------------

    if (
        meshcat is not None
        and grasp_z is not None
    ):

        visualize_detection(
            meshcat,
            estimated_x,
            estimated_y,
            grasp_z,
            points_W,
        )

    print("=" * 55)

    return (
        estimated_x,
        estimated_y,
    )
