"""AR Table Tennis Game.

A single-file Python game that overlays SVG table tennis graphics onto a live
mirrored webcam feed. The player scores points by hitting falling balls with
swift palm swings detected via MediaPipe hand tracking.

All game logic operates in a 1440x1024 reference coordinate space matching the
SVG assets.
"""

import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


class SVGLoader:
    """Load and cache SVG assets as BGRA NumPy arrays.

    Despite the ``.svg`` extension, the asset files are pre-rasterized raster
    buffers decoded via ``cv2.imread`` with ``cv2.IMREAD_UNCHANGED`` so that the
    alpha channel is preserved for fast alpha compositing at runtime.
    """

    # (filename, attribute name) for every asset that must be loaded.
    _ASSETS = (
        ("table-net.svg", "table_net"),
        ("racket-red.svg", "racket_red"),
        ("racket-black.svg", "racket_black"),
        ("ball-white.svg", "ball_white"),
        ("ball-orange.svg", "ball_orange"),
    )

    def __init__(self, assets_dir):
        self.assets_dir = assets_dir
        self.table_net = None    # 1440x1024 BGRA
        self.racket_red = None   # 250x368 BGRA
        self.racket_black = None  # 250x368 BGRA
        self.ball_white = None   # 127x127 BGRA
        self.ball_orange = None  # 127x127 BGRA

    def load_all(self):
        """Load all SVG assets as BGRA NumPy arrays.

        Uses ``cv2.imread(path, cv2.IMREAD_UNCHANGED)`` for each asset. Raises
        ``SystemExit`` with an error message identifying the specific file if any
        asset is missing or cannot be decoded.
        """
        for filename, attr in self._ASSETS:
            path = os.path.join(self.assets_dir, filename)
            if not os.path.exists(path):
                raise SystemExit(
                    "Error: SVG asset not found: {}".format(path)
                )
            image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise SystemExit(
                    "Error: failed to load SVG asset (unreadable or invalid): "
                    "{}".format(path)
                )
            setattr(self, attr, image)

    def get_ball_svg(self, ball_type):
        """Return the ball BGRA buffer matching ``ball_type``.

        ``"white"`` returns ``ball_white`` and ``"orange"`` returns
        ``ball_orange``. Raises ``ValueError`` for any other value.
        """
        if ball_type == "white":
            return self.ball_white
        if ball_type == "orange":
            return self.ball_orange
        raise ValueError("Unknown ball_type: {!r}".format(ball_type))


class CoordinateTransform:
    """Map between the 1440x1024 reference space and actual screen pixels.

    Uses fit-to-contain scaling: the reference rectangle is scaled to fit
    entirely within the window, centered with letterboxing on the axis that
    has excess space.
    """

    REF_WIDTH = 1440
    REF_HEIGHT = 1024

    def __init__(self, window_width, window_height):
        self.window_width = window_width
        self.window_height = window_height
        self._compute_transform()

    def _compute_transform(self):
        """Compute the fit-to-contain scale factor and centering offsets.

        ``scale`` is the smaller of the width and height ratios so the whole
        reference rectangle fits inside the window. ``offset_x`` and
        ``offset_y`` center the scaled rectangle, producing letterboxing on the
        axis with leftover space.
        """
        scale_x = self.window_width / self.REF_WIDTH
        scale_y = self.window_height / self.REF_HEIGHT
        self.scale = min(scale_x, scale_y)
        self.offset_x = (self.window_width - self.REF_WIDTH * self.scale) / 2
        self.offset_y = (self.window_height - self.REF_HEIGHT * self.scale) / 2

    def ref_to_screen(self, rx, ry):
        """Convert reference coordinates to integer screen pixel coordinates."""
        sx = int(rx * self.scale + self.offset_x)
        sy = int(ry * self.scale + self.offset_y)
        return (sx, sy)

    def screen_to_ref(self, sx, sy):
        """Convert screen pixel coordinates back to reference coordinates."""
        rx = (sx - self.offset_x) / self.scale
        ry = (sy - self.offset_y) / self.scale
        return (rx, ry)

    def scale_length(self, ref_length):
        """Scale a length from reference space to screen pixels."""
        return int(ref_length * self.scale)


class LayerCompositor:
    """Composite BGRA layers onto a BGR background using alpha blending.

    All methods are static and mutate the ``background`` array in-place using
    vectorized NumPy operations. The per-pixel blend follows
    ``result = fg_bgr * alpha + bg_bgr * (1 - alpha)`` where
    ``alpha = fg_bgra[:, :, 3:4] / 255.0``.
    """

    @staticmethod
    def alpha_blend(background, foreground_bgra, position, size=None):
        """Composite a BGRA foreground onto a BGR background at ``position``.

        ``background`` is a BGR image and ``foreground_bgra`` is a BGRA image.
        ``position`` is the ``(x, y)`` top-left pixel where the foreground is
        placed on the background. If ``size`` is provided as ``(w, h)``, the
        foreground is resized with ``cv2.resize`` before compositing.

        Boundary clipping is handled so the foreground may extend beyond any
        background edge (including negative ``x``/``y`` positions); only the
        overlapping region is blended. ``background`` is modified in-place.
        """
        if size is not None:
            foreground_bgra = cv2.resize(
                foreground_bgra, size, interpolation=cv2.INTER_AREA
            )

        x, y = position
        fg_h, fg_w = foreground_bgra.shape[:2]
        bg_h, bg_w = background.shape[:2]

        # Overlapping region in background coordinates.
        bx0 = max(x, 0)
        by0 = max(y, 0)
        bx1 = min(x + fg_w, bg_w)
        by1 = min(y + fg_h, bg_h)

        # Nothing to composite if the foreground is fully off-screen.
        if bx0 >= bx1 or by0 >= by1:
            return

        # Corresponding region in foreground coordinates.
        fx0 = bx0 - x
        fy0 = by0 - y
        fx1 = fx0 + (bx1 - bx0)
        fy1 = fy0 + (by1 - by0)

        fg_region = foreground_bgra[fy0:fy1, fx0:fx1]
        bg_region = background[by0:by1, bx0:bx1]

        alpha = fg_region[:, :, 3:4] / 255.0
        fg_bgr = fg_region[:, :, :3]
        bg_region[:] = (fg_bgr * alpha + bg_region * (1 - alpha)).astype(np.uint8)

    @staticmethod
    def composite_full_frame(background, overlay_bgra):
        """Composite a full-frame BGRA overlay onto the BGR background.

        ``overlay_bgra`` must share the background's width and height (used for
        the ``table-net.svg`` overlay). Uses a vectorized NumPy alpha blend and
        modifies ``background`` in-place.
        """
        alpha = overlay_bgra[:, :, 3:4] / 255.0
        fg_bgr = overlay_bgra[:, :, :3]
        background[:] = (fg_bgr * alpha + background * (1 - alpha)).astype(np.uint8)

# ---------------------------------------------------------------------------
# Table Geometry Constants (reference space: 1440x1024)
# ---------------------------------------------------------------------------
# Trapezoid corners of the table surface. The table narrows toward the top
# (the opponent's far edge) and widens toward the bottom (the player's near
# edge), matching the perspective baked into ``table-net.svg``.
TABLE_TOP_LEFT = (396, 292)
TABLE_TOP_RIGHT = (1043, 292)
TABLE_BOTTOM_LEFT = (0, 1000)
TABLE_BOTTOM_RIGHT = (1440, 1000)

# Net region. The net spans vertically between these y-values; y < NET_Y_TOP is
# the Opponent_Zone and y > NET_Y_BOTTOM is the Player_Zone.
NET_Y_TOP = 467
NET_Y_BOTTOM = 622

# Asset sizing ratios relative to the on-screen table overlay width.
BALL_SIZE_RATIO = 0.08      # ball diameter = 8% of table width
RACKET_SIZE_RATIO = 0.15    # racket width = 15% of table width
RACKET_ASPECT = 250 / 368   # racket width / height (native 250x368)


class FallingBall:
    """A virtual ball that descends vertically through the reference space.

    Position and speed are expressed in the 1440x1024 reference coordinate
    space. The ball starts at the table top edge (``y = 292``) and moves
    straight down at a constant speed until it is hit or passes the table
    bottom.
    """

    def __init__(self, x, y=292.0, speed=400, ball_type="white"):
        self.x = x                  # Horizontal position (reference coords)
        self.y = y                  # Vertical position (reference coords)
        self.speed = speed          # Fall speed (reference px/s)
        self.ball_type = ball_type  # "white" or "orange"

    def update(self, dt):
        """Advance the vertical position by ``speed * dt``.

        ``dt`` MUST be pre-clamped by the caller to a maximum of 0.1 seconds to
        prevent the ball from skipping large distances on frame drops.
        """
        self.y += self.speed * dt

    def is_past_table_bottom(self):
        """Return True if the ball has descended past the table bottom edge."""
        return self.y > 1000

    def is_in_player_zone(self):
        """Return True if the ball is in the Player_Zone (below the net bottom)."""
        return self.y > 622

    @property
    def center(self):
        """Return ``(x, y)`` in reference coordinates."""
        return (self.x, self.y)


class GameState(Enum):
    """The eight discrete states of the game's finite state machine.

    An orthogonal ``paused`` flag on :class:`GameContext` freezes the
    time-dependent active states without adding a distinct enum member.
    """

    LICENSE = "license"
    TUTORIAL = "tutorial"
    WAIT_COUNTDOWN = "wait_countdown"
    BALL_FALLING = "ball_falling"
    SWING_DETECT = "swing_detect"
    RESULT = "result"
    GAME_OVER = "game_over"


@dataclass
class GameContext:
    """Mutable game state passed between state handlers.

    Groups every piece of runtime state the state machine reads and mutates so
    handlers can remain pure functions of ``(ctx, dt, input)``.
    """

    state: GameState = GameState.LICENSE
    score: int = 0
    round_duration: float = 180.0
    round_time_remaining: float = 180.0
    ball_speed: int = 400
    balls: List[FallingBall] = field(default_factory=list)
    paused: bool = False

    # License state
    license_accepted: bool = False
    license_overlay_active: bool = False

    # Tutorial state
    tutorial_done: bool = False
    # TutorialSystem is implemented in task 5.2; use a string forward reference
    # so game.py stays importable before that class exists.
    tutorial_system: Optional["TutorialSystem"] = None

    # Countdown sub-state
    countdown_value: Optional[int] = None
    countdown_start_time: Optional[float] = None
    countdown_active: bool = False

    # Visual effects
    flash_effect: Optional[Tuple[str, float]] = None  # (color, expire_timestamp)

    # Ball spawning
    last_spawn_time: float = 0.0
    next_spawn_interval: float = 2.0
    ball_spawn_counter: int = 0  # for color alternation

    # Velocity tracking
    # VelocityBuffer is implemented in task 2.2; use a string forward reference
    # and default to None so game.py stays importable now. Task 6/2.2 will
    # initialize this with a real VelocityBuffer instance.
    velocity_buffer: Optional["VelocityBuffer"] = None

    # Coordinate system
    coord_transform: Optional[CoordinateTransform] = None


# ---------------------------------------------------------------------------
# HandTracker (MediaPipe Tasks API)
# ---------------------------------------------------------------------------
# Module-level functions that load the MediaPipe HandLandmarker, run detection
# on RGB frames, and derive the palm center and 3D palm normal from the 21
# hand landmarks. The game tracks a single hand, so detection is configured
# with ``num_hands=1``.

# Landmark index constants (subset used by the swing/collision pipeline).
LANDMARK_WRIST = 0
LANDMARK_INDEX_MCP = 5
LANDMARK_MIDDLE_MCP = 9
LANDMARK_PINKY_MCP = 17


def create_hand_landmarker(model_path):
    """Create and configure a MediaPipe HandLandmarker.

    Loads the model at ``model_path`` with ``num_hands=1`` and all confidence
    thresholds (detection, presence, tracking) set to 0.5. Raises
    ``SystemExit`` with a clear message if the model file is missing.
    """
    if not os.path.exists(model_path):
        raise SystemExit(
            "Error: hand landmarker model not found: {}. Download it from "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task".format(model_path)
        )
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def detect_landmarks(landmarker, frame_rgb):
    """Detect hand landmarks in an RGB frame.

    Wraps ``frame_rgb`` in a MediaPipe image and runs ``landmarker.detect``.
    Returns the list of 21 landmarks for the single detected hand, or ``None``
    if no hand is detected.
    """
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    detection_result = landmarker.detect(mp_image)
    if not detection_result.hand_landmarks:
        return None
    return detection_result.hand_landmarks[0]


def compute_palm_center(landmarks, frame_width, frame_height):
    """Compute the pixel-coordinate palm center.

    Converts landmark 9 (MIDDLE_FINGER_MCP) from normalized coordinates to
    pixel coordinates as ``(round(lm.x * frame_width), round(lm.y *
    frame_height))``.
    """
    lm = landmarks[LANDMARK_MIDDLE_MCP]
    cx = round(lm.x * frame_width)
    cy = round(lm.y * frame_height)
    return (cx, cy)


def compute_palm_normal(landmarks):
    """Compute the unit 3D palm normal vector.

    Uses the cross product ``(lm5 - lm0) x (lm17 - lm0)`` over the 3D landmark
    coordinates (x, y, z), where lm0 is the wrist, lm5 is INDEX_FINGER_MCP, and
    lm17 is PINKY_MCP. Returns the unit normal as a NumPy array ``[nx, ny,
    nz]``. If the cross product has zero length (degenerate landmarks), returns
    ``[0.0, 0.0, 0.0]`` rather than raising.
    """
    wrist = landmarks[LANDMARK_WRIST]
    index_mcp = landmarks[LANDMARK_INDEX_MCP]
    pinky_mcp = landmarks[LANDMARK_PINKY_MCP]

    p0 = np.array([wrist.x, wrist.y, wrist.z])
    p5 = np.array([index_mcp.x, index_mcp.y, index_mcp.z])
    p17 = np.array([pinky_mcp.x, pinky_mcp.y, pinky_mcp.z])

    normal = np.cross(p5 - p0, p17 - p0)
    length = np.linalg.norm(normal)
    if length == 0:
        return np.array([0.0, 0.0, 0.0])
    return normal / length


# ---------------------------------------------------------------------------
# SwingDetector (velocity buffer + swing classification)
# ---------------------------------------------------------------------------
# Tracks the last few palm-center samples to derive hand velocity, and
# classifies a motion as an active swing when the hand moves fast enough while
# the palm faces the camera. All coordinates and timestamps are in pixel space
# and seconds respectively.

# Camera z-axis vector used for the palm-normal facing check.
CAMERA_Z_AXIS = np.array([0.0, 0.0, -1.0])

# Swing thresholds.
SWING_VELOCITY_THRESHOLD = 200.0   # px/s
SWING_NORMAL_DOT_THRESHOLD = -0.5  # dot(palm_normal, camera z-axis)

# Number of consecutive lost-tracking frames after which the buffer resets.
MAX_CONSECUTIVE_LOST = 3


class VelocityBuffer:
    """Circular buffer of the last 5 ``(position, timestamp)`` samples.

    Used to estimate hand velocity from recent palm-center positions. Tracks a
    ``consecutive_lost`` counter so that transient tracking loss clears the
    history once the hand has been missing for too many frames.
    """

    def __init__(self, max_size=5):
        self.buffer = deque(maxlen=max_size)
        self.consecutive_lost = 0

    def push(self, position, timestamp):
        """Append a ``(position, timestamp)`` sample and reset the lost counter.

        ``position`` is a ``(x, y)`` pixel tuple and ``timestamp`` is a float in
        seconds. Recording a fresh sample means tracking is currently active, so
        ``consecutive_lost`` is reset to 0.
        """
        self.buffer.append((position, timestamp))
        self.consecutive_lost = 0

    def mark_lost(self):
        """Record a frame with no hand detected.

        Increments ``consecutive_lost``; once it reaches
        ``MAX_CONSECUTIVE_LOST`` (3) or more, the sample history is cleared so
        that stale positions cannot produce a spurious swing when tracking
        resumes.
        """
        self.consecutive_lost += 1
        if self.consecutive_lost >= MAX_CONSECUTIVE_LOST:
            self.buffer.clear()

    def compute_velocity(self):
        """Return ``(vx, vy, magnitude)`` from the oldest and newest samples.

        Returns ``None`` if fewer than 3 samples are available or if the elapsed
        time between the oldest and newest sample is zero (guards against
        division by zero). Velocity is computed component-wise as
        ``(newest_position - oldest_position) / elapsed_time`` and ``magnitude``
        is the Euclidean norm of ``(vx, vy)``.
        """
        if len(self.buffer) < 3:
            return None

        oldest_pos, oldest_time = self.buffer[0]
        newest_pos, newest_time = self.buffer[-1]

        elapsed = newest_time - oldest_time
        if elapsed == 0:
            return None

        vx = (newest_pos[0] - oldest_pos[0]) / elapsed
        vy = (newest_pos[1] - oldest_pos[1]) / elapsed
        magnitude = float(np.sqrt(vx * vx + vy * vy))
        return (vx, vy, magnitude)


def classify_swing(velocity_magnitude, palm_normal):
    """Classify a motion as an active swing.

    Returns True if and only if ``velocity_magnitude`` exceeds
    ``SWING_VELOCITY_THRESHOLD`` (200 px/s) AND the palm faces the camera, i.e.
    ``dot(palm_normal, [0, 0, -1])`` is less than
    ``SWING_NORMAL_DOT_THRESHOLD`` (-0.5). Any other combination returns False.
    """
    if velocity_magnitude <= SWING_VELOCITY_THRESHOLD:
        return False
    if np.dot(palm_normal, CAMERA_Z_AXIS) >= SWING_NORMAL_DOT_THRESHOLD:
        return False
    return True


# ---------------------------------------------------------------------------
# CollisionEvaluator (distance gate + swing angle validation)
# ---------------------------------------------------------------------------
# Module-level functions that decide whether an active swing registers a hit on
# a falling ball. A hit requires the palm center to be within a collision
# radius of the ball center AND the swing direction to fall within the forward
# cone (20 deg to 70 deg from horizontal). All coordinates are in the 1440x1024
# reference space.

# Forward-cone bounds, in degrees from horizontal.
HIT_ANGLE_MIN = 20.0
HIT_ANGLE_MAX = 70.0

# Top of the Player_Zone in reference space (below the net bottom edge).
PLAYER_ZONE_TOP_Y = 622


def compute_distance(point_a, point_b):
    """Return the Euclidean distance between two 2D points.

    ``point_a`` and ``point_b`` are ``(x, y)`` tuples in reference coordinates.
    """
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    return float(np.sqrt(dx * dx + dy * dy))


def compute_swing_angle(vx, vy):
    """Return the swing angle in degrees from horizontal, in ``[0, 90]``.

    Computes ``arctan2(|vy|, |vx|)`` converted to degrees. Because the absolute
    values of the velocity components are used, the result is symmetric for
    left-to-right and right-to-left swings (only the elevation from horizontal
    matters, not the direction of travel).
    """
    angle_rad = np.arctan2(abs(vy), abs(vx))
    return float(np.degrees(angle_rad))


def evaluate_hit(palm_center, velocity, ball, collision_radius,
                 angle_min=HIT_ANGLE_MIN, angle_max=HIT_ANGLE_MAX):
    """Return True if the swing hits ``ball``.

    A hit requires both conditions to hold:

    1. The Euclidean distance between ``palm_center`` and ``ball.center`` is
       strictly less than ``collision_radius``.
    2. The swing angle from :func:`compute_swing_angle` lies within
       ``[angle_min, angle_max]`` degrees (the forward cone).

    ``velocity`` may be a ``(vx, vy)`` or ``(vx, vy, magnitude)`` tuple; only the
    first two components are used for the angle. Returns False if either
    condition fails.
    """
    distance = compute_distance(palm_center, ball.center)
    if distance >= collision_radius:
        return False

    vx, vy = velocity[0], velocity[1]
    angle = compute_swing_angle(vx, vy)
    return angle_min <= angle <= angle_max


def evaluate_all_balls(palm_center, velocity, balls, player_zone_top_y=PLAYER_ZONE_TOP_Y):
    """Return the list of Player_Zone balls that register a hit.

    Every ball whose ``y`` position is in the Player_Zone (``ball.y >
    player_zone_top_y``) is evaluated independently via :func:`evaluate_hit`
    with a collision radius derived from the ball's rendered radius plus a 30px
    hand radius. Balls outside the Player_Zone are ignored. Returns the list of
    balls that satisfy both the distance and angle conditions.

    ``velocity`` may be a ``(vx, vy)`` or ``(vx, vy, magnitude)`` tuple.
    """
    hit_balls = []
    for ball in balls:
        if ball.y <= player_zone_top_y:
            continue
        # Ball rendered radius in reference space (diameter = 8% of table
        # width) plus a 30px hand radius, matching the collision model.
        ball_radius = (BALL_SIZE_RATIO * CoordinateTransform.REF_WIDTH) / 2
        collision_radius = ball_radius + 30
        if evaluate_hit(palm_center, velocity, ball, collision_radius):
            hit_balls.append(ball)
    return hit_balls


# ---------------------------------------------------------------------------
# ScoringEvaluator (ray projection + opponent-zone determination)
# ---------------------------------------------------------------------------
# Module-level functions that decide whether a registered hit awards a point.
# When a ball is hit, its swing velocity is projected forward from the hit
# point to approximate where the ball lands on the table plane. A point is
# awarded only when that landing falls inside the Opponent_Zone (the far half
# of the table above the net, within the trapezoid boundaries). All
# coordinates are in the 1440x1024 reference space.

# Vertical bounds of the Opponent_Zone in reference space. The zone spans from
# the table top edge (y = 292) up to, but not including, the net top edge
# (NET_Y_TOP = 467): the half-open interval [292, 467).
OPPONENT_ZONE_Y_MIN = TABLE_TOP_LEFT[1]  # 292 (table top edge)
OPPONENT_ZONE_Y_MAX = NET_Y_TOP          # 467 (net top edge, exclusive)

# Height of the trapezoid from top edge (y=292) to bottom edge (y=1000).
TABLE_TRAPEZOID_HEIGHT = 1000 - OPPONENT_ZONE_Y_MIN  # 708


def compute_trapezoid_bounds_at_y(y):
    """Return the ``(x_left, x_right)`` table bounds at reference height ``y``.

    The table is a trapezoid that narrows toward the top, so its horizontal
    bounds are linearly interpolated between the top edge (``y = 292``) and the
    bottom edge (``y = 1000``)::

        t = (y - 292) / 708
        x_left = 396 * (1 - t)     # 396 at the top, 0 at the bottom
        x_right = 1043 + t * 397   # 1043 at the top, 1440 at the bottom

    ``t`` is clamped to ``[0, 1]`` so out-of-range ``y`` values return the
    nearest edge bounds. At ``y = 292`` the bounds are ``(396, 1043)`` and at
    ``y = 1000`` they are ``(0, 1440)``.
    """
    t = (y - OPPONENT_ZONE_Y_MIN) / TABLE_TRAPEZOID_HEIGHT
    t = max(0.0, min(1.0, t))
    x_left = 396 * (1 - t)
    x_right = 1043 + t * 397
    return (x_left, x_right)


def project_landing_position(hit_point, velocity):
    """Project a landing position from ``hit_point`` along ``velocity``.

    ``hit_point`` is the ``(x, y)`` reference-space position where the ball was
    struck and ``velocity`` is a ``(vx, vy)`` or ``(vx, vy, magnitude)`` tuple
    describing the swing direction. Only the first two components are used.

    The player hits from the Player_Zone (``y > 622``) toward the Opponent_Zone
    (``y < 467``), so a valid scoring hit sends the ball upward, i.e. ``vy < 0``
    in image coordinates. If the swing is not directed upward or the vertical
    velocity is degenerate (``vy >= 0``), the ball cannot be projected into the
    opponent's half; the hit point is returned unchanged so the zone check
    fails.

    For an upward swing the hit point is mirror-projected across the net center
    to approximate the landing::

        net_center_y = (NET_Y_TOP + NET_Y_BOTTOM) / 2
        distance_to_net_center = hit_y - net_center_y
        landing_y = hit_y - distance_to_net_center * 2
        landing_x = hit_x + vx * (distance_to_net_center / abs(vy))

    Returns ``(landing_x, landing_y)`` in reference coordinates.
    """
    hit_x, hit_y = float(hit_point[0]), float(hit_point[1])
    vx, vy = float(velocity[0]), float(velocity[1])

    # Upward-velocity requirement / degenerate-velocity guard: a non-upward or
    # zero vertical velocity cannot carry the ball into the Opponent_Zone.
    if vy >= 0:
        return (hit_x, hit_y)

    net_center_y = (NET_Y_TOP + NET_Y_BOTTOM) / 2.0
    distance_to_net_center = hit_y - net_center_y
    landing_y = hit_y - distance_to_net_center * 2
    landing_x = hit_x + vx * (distance_to_net_center / abs(vy))
    return (landing_x, landing_y)


def is_in_opponent_zone(landing_x, landing_y):
    """Return True if ``(landing_x, landing_y)`` is inside the Opponent_Zone.

    A landing is in the Opponent_Zone if and only if its ``y`` lies in the
    half-open interval ``[292, 467)`` (from the table top edge up to the net top
    edge) AND its ``x`` lies within the interpolated trapezoid bounds at that
    ``y`` (see :func:`compute_trapezoid_bounds_at_y`).
    """
    if landing_y < OPPONENT_ZONE_Y_MIN or landing_y >= OPPONENT_ZONE_Y_MAX:
        return False
    x_left, x_right = compute_trapezoid_bounds_at_y(landing_y)
    return x_left <= landing_x <= x_right


def evaluate_score(hit_point, velocity):
    """Return True if a hit at ``hit_point`` with ``velocity`` awards a point.

    Full scoring pipeline: project the landing position via
    :func:`project_landing_position`, then test it against
    :func:`is_in_opponent_zone`. Returns True only when the projected landing
    falls inside the Opponent_Zone.
    """
    landing_x, landing_y = project_landing_position(hit_point, velocity)
    return is_in_opponent_zone(landing_x, landing_y)


# ---------------------------------------------------------------------------
# LicenseScreen (GPLv3 display + acceptance)
# ---------------------------------------------------------------------------
# Loads the GPLv3 ``LICENSE`` file at startup and renders it two ways: as a
# full-screen acceptance overlay on a dark background (initial LICENSE state)
# and as a semi-transparent read-only panel drawn over the live game frame
# (triggered by the 'A' key during gameplay). Text is word-wrapped to the frame
# width and vertically scrollable via ``scroll_offset``. All rendering uses
# OpenCV drawing primitives on BGR frames.


class LicenseScreen:
    """Handle GPLv3 license display and acceptance.

    The full license text is loaded from the project-root ``LICENSE`` file at
    construction time. :meth:`render` draws the acceptance screen, while
    :meth:`render_overlay` draws a read-only panel over the current game frame.
    ``scroll_offset`` tracks the index of the first visible (wrapped) line and
    is clamped to a valid range during rendering.
    """

    LICENSE_FILE = "LICENSE"

    # Rendering constants (OpenCV putText parameters and layout metrics).
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.4
    _FONT_THICKNESS = 1
    _LINE_HEIGHT = 18            # vertical spacing between text lines (px)
    _TEXT_COLOR = (220, 220, 220)      # BGR light grey license body text
    _INSTRUCTION_COLOR = (0, 255, 255)  # BGR yellow instruction text
    _BG_COLOR = (20, 20, 20)           # BGR dark background fill
    _MARGIN_X = 40               # left/right margin (px)
    _TOP_MARGIN = 40             # top margin for first text line (px)
    _BOTTOM_MARGIN = 60          # reserved band for the instruction (px)
    _SCROLL_STEP = 3             # lines advanced per scroll key press

    def __init__(self):
        self.license_text = ""
        self.accepted = False
        self.scroll_offset = 0
        # Maximum valid scroll_offset for the most recent render; updated each
        # time the text is laid out so handle_input can clamp scrolling.
        self.max_scroll_offset = 0
        self.load_license()

    def load_license(self):
        """Load the GPLv3 text from the project-root ``LICENSE`` file.

        Reads :attr:`LICENSE_FILE` and stores the contents in
        ``self.license_text``. Raises ``SystemExit`` with a clear message if the
        file is missing or unreadable, matching the startup error-handling of
        the other asset loaders.
        """
        if not os.path.exists(self.LICENSE_FILE):
            raise SystemExit(
                "Error: LICENSE file not found: {}. The GPLv3 license text is "
                "required before the game can start.".format(self.LICENSE_FILE)
            )
        try:
            with open(self.LICENSE_FILE, "r", encoding="utf-8") as handle:
                self.license_text = handle.read()
        except OSError as exc:
            raise SystemExit(
                "Error: failed to read LICENSE file {}: {}".format(
                    self.LICENSE_FILE, exc
                )
            )

    def _wrap_text_lines(self, max_width):
        """Return the license text word-wrapped to ``max_width`` pixels.

        Splits ``self.license_text`` on newlines (preserving blank lines) and
        greedily word-wraps each line so that its rendered width, measured with
        ``cv2.getTextSize``, does not exceed ``max_width``. A single word wider
        than ``max_width`` is kept on its own line and left to be clipped by the
        frame edge during rendering.
        """
        wrapped = []
        for raw_line in self.license_text.split("\n"):
            if raw_line == "":
                wrapped.append("")
                continue
            current = ""
            for word in raw_line.split(" "):
                candidate = word if current == "" else current + " " + word
                (text_w, _), _ = cv2.getTextSize(
                    candidate, self._FONT, self._FONT_SCALE, self._FONT_THICKNESS
                )
                if text_w <= max_width or current == "":
                    current = candidate
                else:
                    wrapped.append(current)
                    current = word
            wrapped.append(current)
        return wrapped

    def _clamp_scroll(self, line_count, visible_lines):
        """Clamp ``scroll_offset`` to ``[0, max_scroll_offset]``.

        ``max_scroll_offset`` is the number of wrapped lines that do not fit on
        screen (``line_count - visible_lines``, floored at 0) and is stored on
        the instance so :meth:`handle_input` can bound scroll key presses.
        """
        self.max_scroll_offset = max(0, line_count - visible_lines)
        if self.scroll_offset > self.max_scroll_offset:
            self.scroll_offset = self.max_scroll_offset
        if self.scroll_offset < 0:
            self.scroll_offset = 0

    def render(self, frame):
        """Render the full-screen license acceptance overlay.

        Fills ``frame`` with a dark background, draws the scrollable license
        text word-wrapped to the frame width, and shows a "Press Enter to
        accept" instruction (with scroll/exit hints) in a reserved band at the
        bottom. ``frame`` is modified in-place.
        """
        h, w = frame.shape[:2]
        frame[:] = self._BG_COLOR

        max_text_width = w - 2 * self._MARGIN_X
        lines = self._wrap_text_lines(max_text_width)

        usable_height = h - self._TOP_MARGIN - self._BOTTOM_MARGIN
        visible_lines = max(1, usable_height // self._LINE_HEIGHT)
        self._clamp_scroll(len(lines), visible_lines)

        start = self.scroll_offset
        end = start + visible_lines
        y = self._TOP_MARGIN
        for line in lines[start:end]:
            cv2.putText(
                frame, line, (self._MARGIN_X, y), self._FONT, self._FONT_SCALE,
                self._TEXT_COLOR, self._FONT_THICKNESS, cv2.LINE_AA,
            )
            y += self._LINE_HEIGHT

        # Instruction band at the bottom, drawn over an opaque strip so it
        # never overlaps scrolled body text.
        cv2.rectangle(
            frame, (0, h - self._BOTTOM_MARGIN), (w, h), self._BG_COLOR, -1
        )
        instruction = "Press Enter to accept   (Up/Down to scroll, Esc to exit)"
        (text_w, _), _ = cv2.getTextSize(instruction, self._FONT, 0.5, 1)
        ix = max(self._MARGIN_X, (w - text_w) // 2)
        cv2.putText(
            frame, instruction, (ix, h - 20), self._FONT, 0.5,
            self._INSTRUCTION_COLOR, 1, cv2.LINE_AA,
        )

    def render_overlay(self, frame):
        """Render a read-only license panel over the current game frame.

        Draws a semi-transparent dark panel inset from the frame edges (so the
        live webcam feed remains faintly visible), lays out the scrollable
        license text inside it, and shows a "Press any key to dismiss"
        instruction at the bottom of the panel. Used when the 'A' key is pressed
        during gameplay. ``frame`` is modified in-place.
        """
        h, w = frame.shape[:2]
        px0, py0 = self._MARGIN_X, self._TOP_MARGIN
        px1, py1 = w - self._MARGIN_X, h - self._TOP_MARGIN

        # Semi-transparent dark panel over the live frame.
        overlay = frame.copy()
        cv2.rectangle(overlay, (px0, py0), (px1, py1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        inner_margin = 20
        max_text_width = (px1 - px0) - 2 * inner_margin
        lines = self._wrap_text_lines(max_text_width)

        usable_height = (py1 - py0) - inner_margin - self._LINE_HEIGHT
        visible_lines = max(1, usable_height // self._LINE_HEIGHT)
        self._clamp_scroll(len(lines), visible_lines)

        start = self.scroll_offset
        end = start + visible_lines
        y = py0 + inner_margin + 4
        for line in lines[start:end]:
            cv2.putText(
                frame, line, (px0 + inner_margin, y), self._FONT,
                self._FONT_SCALE, self._TEXT_COLOR, self._FONT_THICKNESS,
                cv2.LINE_AA,
            )
            y += self._LINE_HEIGHT

        instruction = "Press any key to dismiss"
        (text_w, _), _ = cv2.getTextSize(instruction, self._FONT, 0.5, 1)
        ix = max(px0, (w - text_w) // 2)
        cv2.putText(
            frame, instruction, (ix, py1 - 15), self._FONT, 0.5,
            self._INSTRUCTION_COLOR, 1, cv2.LINE_AA,
        )

    def handle_input(self, key):
        """Handle a key press on the license screen.

        Returns ``'accept'`` for Enter (key code 13, also recording acceptance),
        ``'exit'`` for Escape (key code 27), and ``'none'`` otherwise. Up/Down
        arrow keys (and 'w'/'s') scroll the text, updating ``scroll_offset``
        clamped to ``[0, max_scroll_offset]``; scroll keys still return
        ``'none'`` since they neither accept nor exit.
        """
        if key == 13:  # Enter
            self.accepted = True
            return "accept"
        if key == 27:  # Escape
            return "exit"

        # Scroll up: Up arrow (waitKey 82 / waitKeyEx 2490368) or 'w'.
        if key in (82, 2490368, ord("w"), ord("W")):
            self.scroll_offset = max(0, self.scroll_offset - self._SCROLL_STEP)
        # Scroll down: Down arrow (waitKey 84 / waitKeyEx 2621440) or 's'.
        elif key in (84, 2621440, ord("s"), ord("S")):
            self.scroll_offset = min(
                self.max_scroll_offset, self.scroll_offset + self._SCROLL_STEP
            )
        return "none"


# ---------------------------------------------------------------------------
# TutorialSystem (guided 3-ball first-launch tutorial)
# ---------------------------------------------------------------------------
# Runs a short guided sequence the first time the game is launched: a single
# ball descends, pauses under a circular spotlight with an explanatory caption,
# then resumes so the player can try to hit it, followed by two more free
# practice balls and a final "ready to start" prompt. Completion is persisted
# to a ``.tutorial_done`` marker file so the tutorial is skipped on subsequent
# launches. All ball coordinates are in the 1440x1024 reference space; the
# spotlight/instruction rendering operates on screen-space BGR frames.


class TutorialPhase(Enum):
    """The phases of the guided tutorial sequence.

    ``BALL_DESCENDING`` -> ``SPOTLIGHT_PAUSE`` -> ``BALL_RESUMED`` ->
    ``PRACTICE_BALLS`` -> ``COMPLETE``. The first ball drives the descend →
    spotlight → resume steps; the two follow-up practice balls run in
    ``PRACTICE_BALLS``; ``COMPLETE`` shows the start prompt and waits for Enter.
    """

    BALL_DESCENDING = "ball_descending"
    SPOTLIGHT_PAUSE = "spotlight_pause"
    BALL_RESUMED = "ball_resumed"
    PRACTICE_BALLS = "practice_balls"
    COMPLETE = "complete"


class TutorialSystem:
    """Manage the guided 3-ball tutorial sequence.

    The system owns a single active ``tutorial_ball`` at a time and advances a
    small phase machine via :meth:`update`. The caller drives it each frame with
    the frame delta-time and the latest key code, and renders the spotlight and
    instruction text via :meth:`render_spotlight` and
    :meth:`render_instructions`. A ball counts as resolved when it is hit (the
    caller sets ``tutorial_ball`` to ``None``) or when it falls past the table
    bottom.
    """

    TUTORIAL_DONE_FILE = ".tutorial_done"
    SPOTLIGHT_THRESHOLD = 0.4  # fraction of frame height at which the ball pauses

    # Default tutorial ball speed (reference px/s) and geometry.
    TUTORIAL_BALL_SPEED = 400
    TUTORIAL_TABLE_TOP_Y = TABLE_TOP_LEFT[1]  # 292 (table top edge)
    TUTORIAL_TABLE_X_CENTER = (TABLE_TOP_LEFT[0] + TABLE_TOP_RIGHT[0]) / 2.0  # 719.5

    # Maximum delta-time applied to ball physics (matches project convention).
    MAX_DT = 0.1

    # Total number of tutorial balls (1 guided + 2 practice).
    PRACTICE_BALL_COUNT = 2

    # Instruction captions shown during the sequence.
    SPOTLIGHT_MESSAGE = "This is the ball you need to hit"
    COMPLETE_MESSAGE = "Ready to start? Press Enter to begin"

    # Spotlight radius (screen pixels) used by the default caller.
    SPOTLIGHT_RADIUS = 140

    # render_instructions text styling (OpenCV putText parameters).
    _FONT = cv2.FONT_HERSHEY_SIMPLEX
    _FONT_SCALE = 0.9
    _FONT_THICKNESS = 2
    _TEXT_COLOR = (255, 255, 255)   # BGR white body text
    _OUTLINE_COLOR = (0, 0, 0)      # BGR black outline for contrast
    _OUTLINE_THICKNESS = 5

    def __init__(self):
        self.phase = TutorialPhase.BALL_DESCENDING
        self.tutorial_ball = self.spawn_tutorial_ball(
            self.TUTORIAL_TABLE_TOP_Y,
            self.TUTORIAL_TABLE_X_CENTER,
            self.TUTORIAL_BALL_SPEED,
        )
        # Practice balls still to spawn/resolve after the guided ball.
        self.practice_balls_remaining = self.PRACTICE_BALL_COUNT
        # Total balls resolved so far (hit or missed).
        self.balls_resolved = 0

    @staticmethod
    def is_tutorial_done():
        """Return True if the ``.tutorial_done`` marker file exists."""
        return os.path.exists(TutorialSystem.TUTORIAL_DONE_FILE)

    @staticmethod
    def mark_tutorial_done():
        """Create the ``.tutorial_done`` marker file to persist completion.

        Failure to write the marker is non-critical (the tutorial simply repeats
        on the next launch), so any ``OSError`` is swallowed silently rather than
        interrupting the game.
        """
        try:
            with open(TutorialSystem.TUTORIAL_DONE_FILE, "w", encoding="utf-8") as handle:
                handle.write("")
        except OSError:
            # Non-critical: tutorial will simply repeat next launch.
            pass

    def spawn_tutorial_ball(self, table_top_y, table_x_center, speed):
        """Return a :class:`FallingBall` spawned at the center of the table top.

        The ball is placed at ``table_x_center`` horizontally and ``table_top_y``
        vertically (the top edge of the table trapezoid) so it descends straight
        down the middle of the table for the tutorial.
        """
        return FallingBall(
            x=float(table_x_center),
            y=float(table_top_y),
            speed=speed,
            ball_type="white",
        )

    def _ball_resolved(self):
        """Return True if the current tutorial ball has been resolved.

        A ball is resolved when the caller has cleared it after a hit
        (``tutorial_ball is None``) or when it has fallen past the table bottom.
        """
        return self.tutorial_ball is None or self.tutorial_ball.is_past_table_bottom()

    def update(self, dt, key):
        """Advance the tutorial phase machine one frame.

        ``dt`` is the frame delta-time in seconds (clamped internally to
        :attr:`MAX_DT` before it drives ball physics) and ``key`` is the latest
        key code (or -1 when no key was pressed). Returns an action string:

        - ``'continue'`` while a phase is still in progress,
        - ``'advance'`` when the spotlight pause is dismissed with Enter,
        - ``'complete'`` when the player presses Enter at the completion prompt,
        - ``None`` if the phase machine is in an unrecognized state.

        Phase flow: the ball descends until it reaches
        :attr:`SPOTLIGHT_THRESHOLD` of the frame height, pauses under the
        spotlight until Enter, resumes and is resolved, then two practice balls
        are spawned and resolved before the completion prompt.
        """
        dt = min(dt, self.MAX_DT)

        if self.phase == TutorialPhase.BALL_DESCENDING:
            self.tutorial_ball.update(dt)
            threshold_y = self.SPOTLIGHT_THRESHOLD * CoordinateTransform.REF_HEIGHT
            if self.tutorial_ball.y >= threshold_y:
                self.phase = TutorialPhase.SPOTLIGHT_PAUSE
            return "continue"

        if self.phase == TutorialPhase.SPOTLIGHT_PAUSE:
            # Ball movement is frozen; wait for Enter to resume.
            if key == 13:
                self.phase = TutorialPhase.BALL_RESUMED
                return "advance"
            return "continue"

        if self.phase == TutorialPhase.BALL_RESUMED:
            if self.tutorial_ball is not None:
                self.tutorial_ball.update(dt)
            if self._ball_resolved():
                self.balls_resolved += 1
                self._start_next_practice_ball()
            return "continue"

        if self.phase == TutorialPhase.PRACTICE_BALLS:
            if self.tutorial_ball is not None:
                self.tutorial_ball.update(dt)
            if self._ball_resolved():
                self.balls_resolved += 1
                self.practice_balls_remaining -= 1
                if self.practice_balls_remaining > 0:
                    self.tutorial_ball = self.spawn_tutorial_ball(
                        self.TUTORIAL_TABLE_TOP_Y,
                        self.TUTORIAL_TABLE_X_CENTER,
                        self.TUTORIAL_BALL_SPEED,
                    )
                else:
                    self.tutorial_ball = None
                    self.phase = TutorialPhase.COMPLETE
            return "continue"

        if self.phase == TutorialPhase.COMPLETE:
            if key == 13:
                self.mark_tutorial_done()
                return "complete"
            return "continue"

        return None

    def _start_next_practice_ball(self):
        """Transition into the practice-ball phase and spawn the first one.

        Called once the guided (first) ball is resolved. If no practice balls
        are configured, jumps straight to the completion prompt instead.
        """
        if self.practice_balls_remaining > 0:
            self.phase = TutorialPhase.PRACTICE_BALLS
            self.tutorial_ball = self.spawn_tutorial_ball(
                self.TUTORIAL_TABLE_TOP_Y,
                self.TUTORIAL_TABLE_X_CENTER,
                self.TUTORIAL_BALL_SPEED,
            )
        else:
            self.tutorial_ball = None
            self.phase = TutorialPhase.COMPLETE

    def render_spotlight(self, frame, ball_center, radius):
        """Dim the frame everywhere outside a circular spotlight.

        Pixels outside the circle centered on ``ball_center`` (screen pixel
        coordinates) with the given ``radius`` are darkened to 30% brightness
        (a 70% dim), leaving the ball highlighted. ``frame`` is modified
        in-place. A binary mask marks the spotlight interior so only the outside
        region is replaced with the darkened copy.
        """
        dark = (frame * 0.3).astype(np.uint8)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (int(ball_center[0]), int(ball_center[1])), int(radius), 255, -1)
        frame[mask == 0] = dark[mask == 0]

    def render_instructions(self, frame, text, position):
        """Draw tutorial instruction ``text`` at ``position`` on ``frame``.

        Renders the text with a black outline beneath white glyphs so it stays
        legible over both the darkened spotlight background and the live webcam
        feed. ``position`` is the ``(x, y)`` bottom-left baseline of the text in
        screen pixels. ``frame`` is modified in-place.
        """
        cv2.putText(
            frame, text, position, self._FONT, self._FONT_SCALE,
            self._OUTLINE_COLOR, self._OUTLINE_THICKNESS, cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, position, self._FONT, self._FONT_SCALE,
            self._TEXT_COLOR, self._FONT_THICKNESS, cv2.LINE_AA,
        )


# ---------------------------------------------------------------------------
# Ball spawning and physics update (state machine helpers)
# ---------------------------------------------------------------------------
# Module-level functions that drive falling-ball creation and per-frame motion.
# All positions are in the 1440x1024 reference coordinate space; balls spawn
# along the table's top edge (y = 292) and are retired once they fall past the
# table's bottom edge (y = 1000). A miss flash is triggered for each retired
# ball so the player gets visual feedback for balls they failed to return.

# Duration (seconds) that a flash overlay stays active. Kept within the
# 150-250ms window specified by the design's visual-effects convention.
MISS_FLASH_DURATION = 0.2


def compute_dt(prev_time, current_time):
    """Return the frame delta-time clamped to a safe maximum.

    Clamping ``current_time - prev_time`` to at most 0.1 seconds prevents a
    physics "explosion" when the frame rate drops (a large gap would otherwise
    teleport balls far down the screen in a single update).
    """
    raw_dt = current_time - prev_time
    return min(raw_dt, 0.1)


def maybe_spawn_ball(ctx, current_time):
    """Spawn a new falling ball when the spawn interval has elapsed.

    A ball is created only if ``current_time - ctx.last_spawn_time`` has reached
    ``ctx.next_spawn_interval``. New balls appear at a random horizontal position
    along the table's top edge (x in [396, 1043], y = 292) and inherit the
    current ``ctx.ball_speed``. Ball color alternates via ``ctx.ball_spawn_counter``
    ("white" on even counts, "orange" on odd counts). After spawning, the counter
    is incremented, the spawn clock is reset, and the next interval is randomized
    within [1.5, 2.5] seconds.
    """
    if current_time - ctx.last_spawn_time >= ctx.next_spawn_interval:
        # Table top x-bounds in reference space: 396 (left) .. 1043 (right).
        x_left, x_right = TABLE_TOP_LEFT[0], TABLE_TOP_RIGHT[0]
        x = random.uniform(x_left, x_right)
        ball_type = "white" if ctx.ball_spawn_counter % 2 == 0 else "orange"
        ctx.balls.append(
            FallingBall(x=x, y=292.0, speed=ctx.ball_speed, ball_type=ball_type)
        )
        ctx.ball_spawn_counter += 1
        ctx.last_spawn_time = current_time
        ctx.next_spawn_interval = random.uniform(1.5, 2.5)


def update_balls(ctx, dt):
    """Advance every active ball and retire those that fall off the table.

    Each ball's vertical position is advanced by ``dt`` (the caller must supply a
    pre-clamped delta-time via :func:`compute_dt`). Balls whose y-coordinate
    passes the table's bottom edge (y > 1000) are removed from ``ctx.balls`` and
    treated as misses: a red flash effect is armed for each retired ball so the
    player sees feedback for the missed return.
    """
    for ball in ctx.balls:
        ball.update(dt)

    remaining = []
    missed = False
    for ball in ctx.balls:
        if ball.is_past_table_bottom():
            missed = True
        else:
            remaining.append(ball)
    ctx.balls = remaining

    if missed:
        # Arm a red "miss" flash; the render layer compares the expire timestamp
        # against the current clock to decide when to stop drawing it.
        ctx.flash_effect = ("red", time.time() + MISS_FLASH_DURATION)

# ---------------------------------------------------------------------------
# Pause handling and license overlay logic
# ---------------------------------------------------------------------------
# Module-level helpers that implement the game's orthogonal PAUSED behavior and
# the read-only in-game license overlay. Neither pause nor the overlay is a
# distinct state in :class:`GameState`; both are flags on :class:`GameContext`
# that freeze the time-dependent gameplay updates while leaving the underlying
# state intact for seamless resumption. These helpers are pure functions of
# ``(ctx, key)`` (plus a predicate over ``ctx``) so ``tick_state_machine``
# (task 6.1) can dispatch input and gate updates cleanly.

# Key codes (cv2.waitKey values) recognized by the pause/overlay handlers.
KEY_NONE = -1  # cv2.waitKey timeout: no key pressed this frame
KEY_SPACE = 32  # Spacebar toggles pause during an active round
KEY_A_LOWER = 97  # 'a' opens the license overlay
KEY_A_UPPER = 65  # 'A' opens the license overlay

# States during which Spacebar is allowed to toggle the pause flag. In every
# other state (LICENSE, TUTORIAL, WAIT_COUNTDOWN, GAME_OVER) Spacebar is ignored
# and the pause flag is left untouched.
PAUSE_TOGGLE_STATES = (
    GameState.BALL_FALLING,
    GameState.SWING_DETECT,
    GameState.RESULT,
)


def is_gameplay_active(ctx):
    """Return whether time-dependent gameplay updates should run this frame.

    Gameplay is "active" only when neither the pause flag nor the license
    overlay is engaged. When this returns ``False`` the caller
    (:func:`tick_state_machine`) must skip applying ``dt`` to the round timer,
    skip advancing/removing balls, and skip spawning new balls. Hand tracking
    and rendering continue regardless, so the player still sees their live feed
    (and the racket) while paused or while reading the license overlay.
    """
    return not ctx.paused and not ctx.license_overlay_active


def handle_pause_input(ctx, key):
    """Toggle the pause flag in response to Spacebar during an active round.

    Spacebar (``key == 32``) flips ``ctx.paused`` only while ``ctx.state`` is one
    of :data:`PAUSE_TOGGLE_STATES` (BALL_FALLING, SWING_DETECT, RESULT). In every
    other state the key is ignored and ``ctx.paused`` is left unchanged. The
    toggle is also suppressed while the license overlay is active, because in
    that mode any key is consumed to dismiss the overlay (see
    :func:`handle_license_overlay_input`). Returns ``True`` if the pause flag was
    toggled, ``False`` otherwise.
    """
    if key != KEY_SPACE:
        return False
    if ctx.license_overlay_active:
        return False
    if ctx.state not in PAUSE_TOGGLE_STATES:
        return False
    ctx.paused = not ctx.paused
    return True


def handle_license_overlay_input(ctx, key):
    """Open or dismiss the read-only in-game license overlay.

    Behavior depends on whether the overlay is already showing:

    * **Overlay active:** any key press (``key != -1``) dismisses it by setting
      ``ctx.license_overlay_active = False``. Because the overlay gates gameplay
      via :func:`is_gameplay_active` (rather than mutating ``ctx.paused``),
      dismissing it automatically restores whatever gameplay state was in effect
      beforehand -- if the round was paused before the overlay opened it stays
      paused, otherwise play resumes. Returns ``"dismissed"``.
    * **Overlay inactive:** pressing 'A' (``key == 97`` or ``key == 65``) opens
      the overlay by setting ``ctx.license_overlay_active = True``. This pauses
      gameplay for the duration of the overlay through the
      :func:`is_gameplay_active` gate, without disturbing the underlying
      ``ctx.paused`` flag. Returns ``"opened"``.

    Returns ``"none"`` when no overlay action is taken.
    """
    if ctx.license_overlay_active:
        if key != KEY_NONE:
            ctx.license_overlay_active = False
            return "dismissed"
        return "none"
    if key in (KEY_A_LOWER, KEY_A_UPPER):
        ctx.license_overlay_active = True
        return "opened"
    return "none"


# ---------------------------------------------------------------------------
# State machine tick (per-frame dispatch)
# ---------------------------------------------------------------------------
# ``tick_state_machine`` is the single entry point the main loop calls once per
# frame. It advances the game's finite state machine by dispatching to a
# per-state handler after applying the orthogonal concerns that cut across every
# state: hand tracking (velocity buffer), the read-only in-game license overlay,
# the global Escape-to-quit control, and the Spacebar pause toggle.
#
# All game logic operates in the 1440x1024 reference space, so the palm center
# pushed to the velocity buffer is computed against the reference dimensions
# (not the raw webcam frame size). This keeps hand velocity in reference px/s,
# directly comparable to the swing threshold and to ball positions used by the
# collision and scoring evaluators.
#
# The function returns ``'exit'`` when the application should terminate (Escape
# from any state, or any key while in GAME_OVER) and ``None`` otherwise.

# Key codes recognized by the state machine (cv2.waitKey values).
KEY_ENTER = 13    # accept license / advance tutorial / start countdown
KEY_ESCAPE = 27   # quit from any state

# Countdown length: 3 -> 2 -> 1, one second per number (Requirement 8.2).
COUNTDOWN_SECONDS = 3

# Duration (seconds) a hit/no-score flash stays armed. Kept within the
# 150-250ms window from the design's visual-effects convention.
HIT_FLASH_DURATION = 0.2


def tick_state_machine(ctx, dt, landmarks, key, current_time):
    """Advance the game state machine by one frame and return an action signal.

    Parameters
    ----------
    ctx : GameContext
        The mutable game state; updated in-place.
    dt : float
        Frame delta-time in seconds, already clamped by :func:`compute_dt`.
    landmarks : Optional[list]
        The 21 hand landmarks for the tracked hand, or ``None`` when no hand is
        detected this frame.
    key : int
        The latest key code from ``cv2.waitKey`` (``-1`` when no key pressed).
    current_time : float
        The current wall-clock timestamp (``time.time()``), used for countdown
        timing, ball spawning, and flash-effect expiry.

    Returns
    -------
    Optional[str]
        ``'exit'`` if the application should terminate, otherwise ``None``.

    The dispatch order is: (1) update hand tracking regardless of state so the
    velocity buffer stays fresh, (2) service the in-game license overlay (which
    freezes the tick while shown), (3) honor the global Escape-to-quit control,
    (4) apply the Spacebar pause toggle, then (5) run the handler for the current
    :class:`GameState`. Timer expiry (``round_time_remaining <= 0``) during any
    active state overrides all other transitions and forces GAME_OVER.
    """
    # (1) Hand tracking continues regardless of pause, overlay, or state so the
    # velocity buffer is warm the moment a swing must be evaluated. The palm
    # center is expressed in reference space (1440x1024) to match ball
    # coordinates and the swing velocity threshold.
    if ctx.velocity_buffer is not None:
        if landmarks is None:
            ctx.velocity_buffer.mark_lost()
        else:
            palm_center = compute_palm_center(
                landmarks,
                CoordinateTransform.REF_WIDTH,
                CoordinateTransform.REF_HEIGHT,
            )
            ctx.velocity_buffer.push(palm_center, current_time)

    # (2) Read-only in-game license overlay ('A' to open, any key to dismiss).
    # It is not applicable to the initial LICENSE acceptance screen, which has
    # its own full-screen rendering. While the overlay is showing, the whole
    # tick is frozen (no timer, ball, or transition updates) and control returns
    # immediately so gameplay resumes exactly where it left off on dismissal.
    if ctx.state != GameState.LICENSE:
        overlay_action = handle_license_overlay_input(ctx, key)
        if overlay_action in ("opened", "dismissed") or ctx.license_overlay_active:
            return None

    # (3) Global quit: Escape terminates from any state (Requirement 14.1).
    if key == KEY_ESCAPE:
        return "exit"

    # (4) Spacebar pause toggle (only honored during active rounds; ignored in
    # LICENSE/TUTORIAL/WAIT_COUNTDOWN/GAME_OVER by handle_pause_input).
    handle_pause_input(ctx, key)

    # (5) Per-state dispatch.
    state = ctx.state
    if state == GameState.LICENSE:
        return _tick_license(ctx, key)
    if state == GameState.TUTORIAL:
        return _tick_tutorial(ctx, dt, key)
    if state == GameState.WAIT_COUNTDOWN:
        return _tick_wait_countdown(ctx, key, current_time)
    if state == GameState.BALL_FALLING:
        return _tick_ball_falling(ctx, dt, current_time)
    if state == GameState.SWING_DETECT:
        return _tick_swing_detect(ctx, dt, landmarks, current_time)
    if state == GameState.RESULT:
        return _tick_result(ctx, dt, current_time)
    if state == GameState.GAME_OVER:
        return _tick_game_over(ctx, key)
    return None


def _tick_license(ctx, key):
    """Handle the LICENSE acceptance screen.

    Pressing Enter records acceptance and transitions to TUTORIAL on first
    launch (tutorial not yet done) or straight to WAIT_COUNTDOWN on subsequent
    launches. Escape is handled globally in :func:`tick_state_machine` as a
    quit, so it never reaches this handler. Returns ``None`` (no exit signal).
    """
    if key == KEY_ENTER:
        ctx.license_accepted = True
        if ctx.tutorial_done:
            ctx.state = GameState.WAIT_COUNTDOWN
        else:
            ctx.state = GameState.TUTORIAL
    return None


def _tick_tutorial(ctx, dt, key):
    """Handle the guided TUTORIAL state.

    Delegates to :meth:`TutorialSystem.update`, which advances the ball
    descend -> spotlight -> resume -> practice sequence, consumes Enter to
    advance steps, and persists the ``.tutorial_done`` marker when the player
    presses Enter at the completion prompt (returning ``'complete'``). On
    completion this handler marks the tutorial done in the context and
    transitions to WAIT_COUNTDOWN. Returns ``None``.
    """
    if ctx.tutorial_system is None:
        # No tutorial configured: skip straight to the countdown wait.
        ctx.tutorial_done = True
        ctx.state = GameState.WAIT_COUNTDOWN
        return None

    action = ctx.tutorial_system.update(dt, key)
    if action == "complete":
        ctx.tutorial_done = True
        ctx.state = GameState.WAIT_COUNTDOWN
    return None


def _tick_wait_countdown(ctx, key, current_time):
    """Handle the WAIT_COUNTDOWN state (speed selection + 3-2-1 countdown).

    While the countdown is not yet active the player adjusts the speed trackbar
    (read by the main loop) and presses Enter to start the countdown, which arms
    ``countdown_active``, seeds ``countdown_value`` to 3, and records
    ``countdown_start_time``. Once active, the displayed value is derived from
    the elapsed wall-clock time (one second per number); when the full
    :data:`COUNTDOWN_SECONDS` has elapsed the round begins: the state moves to
    BALL_FALLING, the round timer is (re)initialized to ``round_duration``, and
    ball-spawn timing is primed so the first ball is not spawned instantly.
    Returns ``None``.
    """
    if not ctx.countdown_active:
        if key == KEY_ENTER:
            ctx.countdown_active = True
            ctx.countdown_value = COUNTDOWN_SECONDS
            ctx.countdown_start_time = current_time
        return None

    elapsed = current_time - ctx.countdown_start_time
    if elapsed >= COUNTDOWN_SECONDS:
        # Countdown finished: start the round.
        ctx.countdown_active = False
        ctx.countdown_value = None
        ctx.countdown_start_time = None
        ctx.round_time_remaining = ctx.round_duration
        ctx.last_spawn_time = current_time
        ctx.next_spawn_interval = random.uniform(1.5, 2.5)
        ctx.state = GameState.BALL_FALLING
    else:
        # Display 3 during [0,1), 2 during [1,2), 1 during [2,3).
        ctx.countdown_value = COUNTDOWN_SECONDS - int(elapsed)
    return None


def _advance_round_timer(ctx, dt):
    """Decrement the round timer by ``dt`` and force GAME_OVER on expiry.

    Subtracts the clamped frame delta-time from ``round_time_remaining`` (floored
    at 0). When the timer reaches 0 the state is switched to GAME_OVER and
    ``True`` is returned so the caller can stop further per-frame work; otherwise
    returns ``False``. This centralizes the timer-expiry override shared by the
    active states (BALL_FALLING, SWING_DETECT, RESULT).
    """
    ctx.round_time_remaining = max(0.0, ctx.round_time_remaining - dt)
    if ctx.round_time_remaining <= 0:
        ctx.state = GameState.GAME_OVER
        return True
    return False


def _tick_ball_falling(ctx, dt, current_time):
    """Handle the BALL_FALLING state.

    When gameplay is active (not paused, no overlay) the round timer advances,
    balls fall and retire past the table bottom via :func:`update_balls`, and new
    balls spawn via :func:`maybe_spawn_ball`. Timer expiry overrides everything
    and forces GAME_OVER. If any active ball has entered the Player_Zone
    (``y > 622``) the state transitions to SWING_DETECT so the swing can be
    evaluated. While paused or while the overlay is shown, no time-dependent
    update runs. Returns ``None``.
    """
    if not is_gameplay_active(ctx):
        return None
    if _advance_round_timer(ctx, dt):
        return None
    update_balls(ctx, dt)
    maybe_spawn_ball(ctx, current_time)
    if any(ball.is_in_player_zone() for ball in ctx.balls):
        ctx.state = GameState.SWING_DETECT
    return None


def _tick_swing_detect(ctx, dt, landmarks, current_time):
    """Handle the SWING_DETECT state.

    Physics continues exactly as in BALL_FALLING (timer, ball motion, spawning,
    miss handling) so balls are never frozen while waiting for a swing. When a
    hand is tracked the swing velocity is computed from the velocity buffer and,
    if the motion classifies as an active swing (fast enough and palm facing the
    camera), every Player_Zone ball is evaluated independently via
    :func:`evaluate_all_balls`. If one or more balls are hit, the hit balls and
    the swing velocity are stashed on the context for the RESULT state to score
    and the state transitions to RESULT. If no hit occurs and no ball remains in
    the Player_Zone, the state returns to BALL_FALLING. Timer expiry forces
    GAME_OVER. Returns ``None``.

    The pending hit balls and velocity are stored as ``_pending_hits`` and
    ``_pending_velocity`` attributes so they survive the one-frame gap between
    detecting the hit here and scoring it in :func:`_tick_result`.
    """
    if not is_gameplay_active(ctx):
        return None
    if _advance_round_timer(ctx, dt):
        return None
    update_balls(ctx, dt)
    maybe_spawn_ball(ctx, current_time)

    hit_balls = []
    velocity = None
    if landmarks is not None and ctx.velocity_buffer is not None:
        vel_result = ctx.velocity_buffer.compute_velocity()
        if vel_result is not None:
            vx, vy, magnitude = vel_result
            palm_normal = compute_palm_normal(landmarks)
            if classify_swing(magnitude, palm_normal):
                palm_center = compute_palm_center(
                    landmarks,
                    CoordinateTransform.REF_WIDTH,
                    CoordinateTransform.REF_HEIGHT,
                )
                velocity = (vx, vy)
                hit_balls = evaluate_all_balls(palm_center, velocity, ctx.balls)

    if hit_balls:
        ctx._pending_hits = hit_balls
        ctx._pending_velocity = velocity
        ctx.state = GameState.RESULT
        return None

    # No hit this frame: keep detecting while balls remain in the Player_Zone,
    # otherwise fall back to spawning/falling.
    if not any(ball.is_in_player_zone() for ball in ctx.balls):
        ctx.state = GameState.BALL_FALLING
    return None


def _tick_result(ctx, dt, current_time):
    """Handle the RESULT state (scoring + flash) then return to BALL_FALLING.

    For each ball hit during SWING_DETECT the scoring pipeline
    (:func:`evaluate_score`) projects the swing to a landing position; a point is
    awarded when the landing falls inside the Opponent_Zone. Every evaluated ball
    is removed from the active list. A flash effect is armed for feedback: green
    when at least one point was scored, yellow when balls were hit but none
    scored (matching Requirements 11.4/11.5 and the design's flash convention;
    red flashes are reserved for missed balls that fall past the table bottom and
    are handled in :func:`update_balls`). The timer still advances during this
    frame and expiry forces GAME_OVER; otherwise the state returns to
    BALL_FALLING. Returns ``None``.
    """
    if not is_gameplay_active(ctx):
        return None
    if _advance_round_timer(ctx, dt):
        return None

    hit_balls = getattr(ctx, "_pending_hits", [])
    velocity = getattr(ctx, "_pending_velocity", None)

    scored = False
    for ball in hit_balls:
        if velocity is not None and evaluate_score(ball.center, velocity):
            ctx.score += 1
            scored = True
        if ball in ctx.balls:
            ctx.balls.remove(ball)

    if hit_balls:
        color = "green" if scored else "yellow"
        ctx.flash_effect = (color, current_time + HIT_FLASH_DURATION)

    # Clear the pending hit hand-off and resume falling.
    ctx._pending_hits = []
    ctx._pending_velocity = None
    ctx.state = GameState.BALL_FALLING
    return None


def _tick_game_over(ctx, key):
    """Handle the GAME_OVER state.

    The final score display is drawn by the renderer; this handler only waits
    for input. Any key press (``key != -1``) terminates the application by
    returning ``'exit'``. Escape is already handled as a global quit in
    :func:`tick_state_machine`. Returns ``None`` while waiting.
    """
    if key != KEY_NONE:
        return "exit"
    return None
