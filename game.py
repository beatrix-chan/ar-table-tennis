"""AR Table Tennis Game.

A single-file Python game that overlays SVG table tennis graphics onto a live
mirrored webcam feed. The player scores points by hitting falling balls with
swift palm swings detected via MediaPipe hand tracking.

All game logic operates in a 1440x1024 reference coordinate space matching the
SVG assets.
"""

import os
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
