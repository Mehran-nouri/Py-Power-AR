import cv2
import mediapipe as mp
import numpy as np
import math
import random
import time
import os

# ==========================================
# 1. Settings
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    model_complexity=1
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

# ==========================================
# 2. One Euro Filter
# ==========================================
class OneEuroFilter:
    def __init__(self, min_cutoff=1.9, beta=0.09, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = self.raw_prev = self.dx_prev = None
        self.speed_smooth = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t=None, shared_speed=None):
        if t is None:
            t = time.perf_counter()
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = self.raw_prev = x
            self.dx_prev = 0
            return x
        dt = max(t - self.t_prev, 0.001)
        raw_dx = (x - self.raw_prev) / dt
        self.raw_prev = x
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * raw_dx + (1 - a_d) * self.dx_prev
        speed = shared_speed if shared_speed is not None else abs(dx_hat)
        self.speed_smooth += (speed - self.speed_smooth) * 0.45
        # Adaptive cutoff: stronger smoothing at low speed, less lag at high speed.
        # Keep the beta contribution bounded so pixel-per-second values don't
        # push the cutoff to an excessively high value.
        speed_norm = min(1.0, self.speed_smooth / 1000.0)
        beta_boost = self.beta * min(self.speed_smooth, 350.0) * 0.015
        cutoff = self.min_cutoff + (7.0 - self.min_cutoff) * speed_norm + beta_boost
        cutoff = max(self.min_cutoff, min(8.5, cutoff))
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

filter_x = OneEuroFilter()
filter_y = OneEuroFilter()
filter_grab_x = OneEuroFilter(2.4, 0.12)
filter_grab_y = OneEuroFilter(2.4, 0.12)

prev_raw_x = prev_raw_y = prev_t = None
shared_speed = 0.0
undo_latched = pinch_latched = False

# ==========================================
# 3. Core State
# ==========================================
prev_x = prev_y = 0
was_drawing = False
stroke_points_buffer = []
BUFFER_MAX = 8
lost_drawing_frames = 0
MAX_LOST_DRAWING_FRAMES = 8
stroke_start_frames = 0
STROKE_START_DURATION = 4

stroke_base_thickness = 4
stroke_current_thickness = 4.0
thickness_smoothing = 0.18

colors = [
    (255, 100, 255), (0, 255, 255), (0, 255, 100),
    (255, 200, 0), (255, 0, 200), (0, 100, 255),
]
color_index = 0
current_object_color = None

trail_points = []
TRAIL_MAX_LIFE = 8

objects = []
current_object_mask = None
grabbed_object_id = -1
grab_anchor_local_x = grab_anchor_local_y = 0
grab_velocity_x = grab_velocity_y = 0
grab_angular_velocity = 0.0
grab_last_rotation = 0.0
grab_last_offset_x = grab_last_offset_y = 0

# Grab tuning
spring_stiffness = 0.42
spring_damping = 0.78
pinch_start_dist = None
pinch_active = False
grab_scale_at_pinch = 1.0
# Adaptive Grab: slower hand = steadier, faster hand = tighter follow
GRAB_MIN_GAIN = 0.80
GRAB_MAX_GAIN = 0.985
GRAB_DEADZONE = 0.22
GRAB_MICRO_SMOOTH = 0.13
GRAB_RESPONSE_BOOST = 0.08
# V12 natural-contact tuning: tiny predictive lead reduces the "rubber band" feel.
GRAB_PREDICT_MIN = 0.06
GRAB_PREDICT_MAX = 0.24
# Pinch hysteresis prevents accidental resize flicker
PINCH_START = 34.0
PINCH_RELEASE = 48.0
PINCH_SCALE_SMOOTH = 0.34
PINCH_ROT_DEADZONE = 1.25
PINCH_ROT_SMOOTH = 0.32

# Grab rotation state
grab_initial_hand_angle = None
grab_initial_object_rotation = 0.0

stable_mode = last_stable_mode = "NONE"
mode_history = []
MODE_HISTORY_SIZE = 4
mode_stable_count = 0
STABLE_THRESHOLD = 2
drawing_hold_frames = 0
DRAWING_HOLD_MIN = 6

inertia_objects = []
nearby_object_ids = set()
magnet_target_id = -1

hand_color_current = np.array([0.0, 200.0, 255.0])
MODE_HAND_COLORS = {
    "DRAWING": (255, 100, 255),
    "GRAB": (0, 255, 255),
    "CANCEL": (0, 165, 255),
    "UNDO": (180, 80, 255),
    "NONE": (0, 200, 255),
}

ripples = []
mode_pulse_strength = 0.0
pred_vx = pred_vy = 0.0

straight_dir = None
straight_strength = 0.0

PEN_STYLES = ["Normal", "Marker", "Neon", "Sketch", "Fountain"]
pen_style_index = 0
USE_SNAP = True
gravity_enabled = True
GRAVITY = 0.42
FLOOR_FRICTION = 0.82
AIR_DRAG = 0.992
BOUNCE = 0.42
OBJECT_BOUNCE = 0.34
OBJECT_FRICTION = 0.88
COLLISION_PUSH = 0.55
ANGULAR_DAMPING = 0.965
GROUND_ANGULAR_DAMPING = 0.82
MAX_PHYSICS_SPEED = 28.0
MAX_ANGULAR_SPEED = 18.0

# V12 — natural interaction polish
COLLISION_RESTITUTION_MIN = 0.18
COLLISION_RESTITUTION_MAX = 0.58
COLLISION_FRICTION = 0.72
COLLISION_TORQUE = 0.055
IMPACT_EFFECT_SPEED = 5.0


# ==========================================
# ✨ UNDO STACK بهینه — فقط diff ذخیره می‌شه
# ==========================================
undo_stack = []           # هر آیتم: {"type": "add"|"remove"|"move", ...}
MAX_UNDO_STEPS = 20
last_shape_name = ""

# ⭐ Latch برای bring_to_front
bring_front_latched = False

PICKER_X, PICKER_Y, PICKER_W, PICKER_H = 20, 180, 36, 280

HAND_VISUAL_SCALE = 0.55

# ==========================================
# 4. Particles & Ripples
# ==========================================
particles = []

def spawn_particles(x, y, color, count=3, speed_range=(0.5, 1.8), life_range=(6, 12)):
    for _ in range(count):
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(*speed_range)
        life = random.randint(*life_range)
        particles.append([x, y, math.cos(ang)*spd, math.sin(ang)*spd, life, life, color])

def update_particles():
    global particles
    particles = [[p[0]+p[2], p[1]+p[3], p[2]*0.91, p[3]*0.91, p[4]-1, p[5], p[6]]
                 for p in particles if p[4] > 1]

def draw_particles(img):
    for p in particles:
        a = p[4] / p[5]
        r = max(1, int(2.0 * a))
        cv2.circle(img, (int(p[0]), int(p[1])), r, p[6], -1)
        if r >= 2:
            cv2.circle(img, (int(p[0]), int(p[1])), max(1, r//2), (255, 255, 255), -1)

def spawn_ripple(x, y, color, intensity=1.0):
    max_r = int(36 * intensity)
    life = int(11 * intensity)
    ripples.append([x, y, 3.0, float(max_r), life, life, color])

def update_ripples():
    global ripples
    new = []
    for r in ripples:
        r[2] += (r[3] - r[2]) * 0.18
        r[4] -= 1
        if r[4] > 0:
            new.append(r)
    ripples = new

def draw_ripples(img):
    for r in ripples:
        a = r[4] / r[5]
        radius = int(r[2])
        col = tuple(int(c * a) for c in r[6])
        cv2.circle(img, (int(r[0]), int(r[1])), radius, col, 1, cv2.LINE_AA)

def spawn_smart_trail(x, y, speed, color):
    if speed < 80:
        count, sr = 1, (0.15, 0.6)
    elif speed < 350:
        count, sr = 2, (0.4, 1.2)
    else:
        count, sr = 3, (0.8, 2.0)
    for _ in range(count):
        ang = random.uniform(0, 2 * math.pi)
        spd = random.uniform(*sr)
        life = random.randint(5, 11)
        particles.append([x, y, math.cos(ang)*spd, math.sin(ang)*spd, life, life, color])

# ==========================================
# 5. Pen + Smart stroke
# ==========================================
def get_pen_params(style, speed_factor):
    if style == "Normal":
        return {"thickness": 4, "jitter": 0}
    if style == "Marker":
        return {"thickness": 9, "jitter": 0}
    if style == "Neon":
        return {"thickness": 3, "jitter": 0}
    if style == "Sketch":
        return {"thickness": 3, "jitter": 1}
    if style == "Fountain":
        return {"thickness": max(2, 6 - int(3 * speed_factor)), "jitter": 0}
    return {"thickness": 4, "jitter": 0}

# ==========================================
# ✨ SMART SMOOTH — باگ عقب‌موندگی رفع شد
# ==========================================
def smart_smooth_point(points, new_pt, speed):
    global straight_dir, straight_strength

    # ⭐ در سرعت بالا، هیچ smoothing نداشته باش
    if speed > 700:
        straight_strength = 0.0
        straight_dir = None
        return new_pt

    if len(points) < 3:
        straight_dir = None
        straight_strength = 0.0
        return new_pt

    dirs = []
    for i in range(max(0, len(points) - 4), len(points) - 1):
        dx = points[i + 1][0] - points[i][0]
        dy = points[i + 1][1] - points[i][1]
        ln = math.hypot(dx, dy)
        if ln > 0.5:
            dirs.append((dx / ln, dy / ln))

    if not dirs:
        return new_pt

    avg_dx = sum(d[0] for d in dirs) / len(dirs)
    avg_dy = sum(d[1] for d in dirs) / len(dirs)
    avg_len = math.hypot(avg_dx, avg_dy)
    if avg_len < 0.01:
        return new_pt
    avg_dx /= avg_len
    avg_dy /= avg_len

    consistency = 0.0
    for d in dirs:
        consistency += max(0.0, d[0] * avg_dx + d[1] * avg_dy)
    consistency /= len(dirs)

    last = points[-1]
    to_new = (new_pt[0] - last[0], new_pt[1] - last[1])
    to_len = math.hypot(*to_new)
    if to_len < 0.5:
        return new_pt

    move_dir = (to_new[0] / to_len, to_new[1] / to_len)
    align = move_dir[0] * avg_dx + move_dir[1] * avg_dy

    if consistency > 0.92 and align > 0.85:
        straight_strength = min(1.0, straight_strength + 0.12)
        straight_dir = (avg_dx, avg_dy)
        proj = to_new[0] * avg_dx + to_new[1] * avg_dy
        # ⭐ blend کمتر = عقب‌موندگی کمتر
        blend = 0.40 + 0.30 * straight_strength
        sx = last[0] + (avg_dx * proj) * blend + to_new[0] * (1 - blend)
        sy = last[1] + (avg_dy * proj) * blend + to_new[1] * (1 - blend)
        # ⭐ مقدار کم برای نرمی، ولی نه عقب‌موندگی
        sx = last[0] * 0.10 + sx * 0.90
        sy = last[1] * 0.10 + sy * 0.90
        return (int(round(sx)), int(round(sy)))
    else:
        straight_strength = max(0.0, straight_strength - 0.08)
        if straight_strength < 0.15:
            straight_dir = None
        sx = last[0] * 0.22 + new_pt[0] * 0.78
        sy = last[1] * 0.22 + new_pt[1] * 0.78
        return (int(round(sx)), int(round(sy)))

# ==========================================
# 6. Helpers
# ==========================================
def get_finger_states(lm, handedness="Right"):
    fingers = [lm[tip].y < lm[pip].y - 0.02 for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]]
    thumb_open = (lm[4].x < lm[3].x - 0.01) if handedness == "Right" else (lm[4].x > lm[3].x + 0.01)
    return fingers, thumb_open

def get_scaled_hand_point(lm, index, w, h):
    pts = [(float(p.x * w), float(p.y * h)) for p in lm]
    palm = [0, 5, 9, 13, 17]
    cx = sum(pts[i][0] for i in palm) / 5
    cy = sum(pts[i][1] for i in palm) / 5
    x, y = pts[index]
    if index == 8:
        return int(x), int(y)
    return int(cx + (x - cx) * HAND_VISUAL_SCALE), int(cy + (y - cy) * HAND_VISUAL_SCALE)

def draw_neon_hand(image, hand_landmarks, w, h, neon_color):
    glow = np.zeros_like(image)
    for a, b in mp_hands.HAND_CONNECTIONS:
        x1, y1 = get_scaled_hand_point(hand_landmarks.landmark, a, w, h)
        x2, y2 = get_scaled_hand_point(hand_landmarks.landmark, b, w, h)
        cv2.line(glow, (x1, y1), (x2, y2), neon_color, 2)
        cv2.line(image, (x1, y1), (x2, y2), neon_color, 1)
    glow = cv2.GaussianBlur(glow, (5, 5), 0)
    cv2.addWeighted(image, 1.0, glow, 0.30, 0, image)
    for i in range(21):
        x, y = get_scaled_hand_point(hand_landmarks.landmark, i, w, h)
        if i == 8:
            cv2.circle(image, (x, y), 3, neon_color, -1)
            cv2.circle(image, (x, y), 1, (255, 255, 255), -1)
        elif i in (4, 12, 16, 20):
            cv2.circle(image, (x, y), 2, neon_color, -1)
        else:
            cv2.circle(image, (x, y), 1, neon_color, -1)

def compute_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

def get_object_center(obj):
    b = obj.get("bbox", [0, 0, 10, 10])
    return obj["offset"][0] + (b[0] + b[2]) * 0.5, obj["offset"][1] + (b[1] + b[3]) * 0.5

def get_hand_distance(lm, w, h):
    # Approximate depth from apparent palm size:
    # larger wrist->middle distance ~= hand closer to camera.
    dist = math.hypot((lm[0].x - lm[9].x) * w, (lm[0].y - lm[9].y) * h)
    return max(0.0, min(1.0, (dist - 40) / 80))

def get_transformed_anchor(obj, local_x, local_y):
    """Return the world-space position of a local grab point after scale/rotation."""
    bbox = obj.get("bbox", [0, 0, 0, 0])
    cxr = (bbox[0] + bbox[2]) * 0.5
    cyr = (bbox[1] + bbox[3]) * 0.5
    scale = float(obj.get("scale", 1.0))
    angle = math.radians(float(obj.get("rotation", 0.0)))
    dx = (local_x - cxr) * scale
    dy = (local_y - cyr) * scale
    ca, sa = math.cos(angle), math.sin(angle)
    # Same transform convention used by cv2.getRotationMatrix2D.
    tx = cxr + ca * dx + sa * dy
    ty = cyr - sa * dx + ca * dy
    return tx, ty

def validate_grab_state():
    global grabbed_object_id, pinch_start_dist, pinch_active
    global grab_initial_hand_angle, pinch_initial_angle
    if not (0 <= grabbed_object_id < len(objects)):
        grabbed_object_id = -1
        pinch_start_dist = None
        pinch_active = False
        grab_initial_hand_angle = None
        pinch_initial_angle = None
        return False
    return True

def find_nearest_object(x, y, max_distance=8):
    best_id, best_dist, best_anchor = -1, float("inf"), (0, 0)
    r = max_distance
    for i in range(len(objects) - 1, -1, -1):
        obj = objects[i]
        mask, offset = obj["mask"], obj["offset"]

        # ⭐ محافظت از bbox خالی
        bbox = obj.get("bbox", None)
        if bbox is None or (bbox[2] - bbox[0] <= 0) or (bbox[3] - bbox[1] <= 0):
            bbox = compute_bbox(mask)
            obj["bbox"] = bbox

        if (x < bbox[0] + offset[0] - r or x > bbox[2] + offset[0] + r or
            y < bbox[1] + offset[1] - r or y > bbox[3] + offset[1] + r):
            continue
        lx, ly = x - offset[0], y - offset[1]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                nx, ny = lx + dx, ly + dy
                # offset در Grab می‌تواند اعشاری باشد؛ NumPy برای index عدد صحیح می‌خواهد.
                ix, iy = int(round(nx)), int(round(ny))
                if 0 <= ix < mask.shape[1] and 0 <= iy < mask.shape[0] and mask[iy, ix] > 0:
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < best_dist:
                        best_dist, best_id, best_anchor = d, i, (ix, iy)
    return best_id, best_anchor[0], best_anchor[1]

def update_nearby_and_magnet(x, y, radius=65):
    global nearby_object_ids, magnet_target_id
    nearby_object_ids = set()
    magnet_target_id = -1
    best_d = radius
    for i, obj in enumerate(objects):
        cx, cy = get_object_center(obj)
        d = math.hypot(x - cx, y - cy)
        if d < radius:
            nearby_object_ids.add(i)
            if d < best_d:
                best_d, magnet_target_id = d, i

def apply_magnet(obj_id, cursor_x, cursor_y, strength=0.12):
    if obj_id < 0 or obj_id >= len(objects):
        return
    obj = objects[obj_id]
    cx, cy = get_object_center(obj)
    dx, dy = cursor_x - cx, cursor_y - cy
    dist = math.hypot(dx, dy)
    if dist < 8 or dist > 70:
        return
    pull = strength * (1.0 - dist / 70.0)
    obj["offset"][0] += int(dx * pull)
    obj["offset"][1] += int(dy * pull)

def render_object(canvas, obj, highlight=False, pulse=0.0, is_nearby=False):
    mask, color, offset = obj["mask"], obj["color"], obj["offset"]
    scale = obj.get("scale", 1.0)
    rotation = obj.get("rotation", 0.0)
    h, w = canvas.shape[:2]
    if offset[0] >= w or offset[0] + mask.shape[1] <= 0:
        return
    if offset[1] >= h or offset[1] + mask.shape[0] <= 0:
        return

    work = mask

    # ⭐ محاسبه bbox دقیق
    bbox = obj.get("bbox", None)
    if bbox is None or (bbox[2] - bbox[0] <= 0):
        bbox = compute_bbox(mask)
        obj["bbox"] = bbox

    needs_transform = (abs(scale - 1.0) > 0.008) or (abs(rotation) > 0.5)

    if needs_transform:
        cxr = (bbox[0] + bbox[2]) * 0.5
        cyr = (bbox[1] + bbox[3]) * 0.5
        M = cv2.getRotationMatrix2D((cxr, cyr), rotation, scale)
        work = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST)

    M = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    shifted = cv2.warpAffine(work, M, (w, h), flags=cv2.INTER_LINEAR)

    boost = 1.35 if highlight else (1.0 + 0.22 * pulse if is_nearby else 1.0)
    display = tuple(min(255, int(c * boost)) for c in color)

    if highlight or is_nearby:
        k = 4 if highlight else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dil = cv2.dilate(shifted, kernel, iterations=1)
        outline = cv2.subtract(dil, shifted)
        ob = 1.5 if highlight else (1.15 + 0.35 * pulse)
        canvas[outline > 0] = tuple(min(255, int(c * ob)) for c in color)

    canvas[shifted > 0] = display

def shift_color(color, amount=30):
    hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0][0]
    hh = (int(hsv[0]) + amount) % 180
    s = min(255, max(180, int(hsv[1]) + 10))
    v = min(255, max(200, int(hsv[2]) + 10))
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hh, s, v]]]), cv2.COLOR_HSV2BGR)[0][0])

def get_hand_angle(lm):
    """زاویه کف دست؛ مستقل‌تر از حرکت خود دست برای چرخاندن شیء."""
    dx = (lm[9].x - lm[0].x)
    dy = (lm[9].y - lm[0].y)
    if abs(dx) < 1e-5 and abs(dy) < 1e-5:
        return None
    return math.degrees(math.atan2(dy, dx))

def angle_delta_deg(a, b):
    """اختلاف زاویه در بازه -180 تا +180."""
    if a is None or b is None:
        return 0.0
    return (a - b + 180.0) % 360.0 - 180.0

def get_pinch_angle(lm):
    """زاویه خط بین شست و اشاره؛ برای چرخش دقیق هنگام Pinch."""
    dx = lm[8].x - lm[4].x
    dy = lm[8].y - lm[4].y
    if abs(dx) < 1e-5 and abs(dy) < 1e-5:
        return None
    return math.degrees(math.atan2(dy, dx))


def angle_between(v1, v2):
    l1, l2 = math.hypot(*v1), math.hypot(*v2)
    if l1 < 0.5 or l2 < 0.5:
        return 180.0
    dot = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (l1 * l2)))
    return math.degrees(math.acos(dot))

def draw_quadratic_segment(mask, p0, p1, p2, thickness):
    t = max(1, int(thickness))
    v1 = (p1[0] - p0[0], p1[1] - p0[1])
    v2 = (p2[0] - p1[0], p2[1] - p1[1])
    if angle_between(v1, v2) < 150:
        cv2.line(mask, p1, p2, 255, t, cv2.LINE_AA)
        cv2.circle(mask, p2, max(1, t // 2), 255, -1, cv2.LINE_AA)
        return
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if length < 1.0:
        return
    d_len = math.hypot(*v1)
    if d_len < 0.5:
        cv2.line(mask, p1, p2, 255, t, cv2.LINE_AA)
        return
    cd = min(length * 0.35, d_len * 0.30, 16.0)
    ctrl = (int(p1[0] + v1[0] / d_len * cd), int(p1[1] + v1[1] / d_len * cd))
    steps = max(3, min(16, int(length / 2.0)))
    prev = p1
    for i in range(1, steps + 1):
        u = i / steps
        inv = 1 - u
        x = inv*inv*p1[0] + 2*inv*u*ctrl[0] + u*u*p2[0]
        y = inv*inv*p1[1] + 2*inv*u*ctrl[1] + u*u*p2[1]
        pt = (int(round(x)), int(round(y)))
        cv2.line(mask, prev, pt, 255, t, cv2.LINE_AA)
        prev = pt
    cv2.circle(mask, p2, max(1, t // 2), 255, -1, cv2.LINE_AA)

def draw_stroke_local(mask, points, thickness, jitter=0):
    n = len(points)
    if n < 2:
        return
    t = max(1, int(thickness))
    if jitter > 0:
        for off in range(jitter):
            jx, jy = random.randint(-1, 1), random.randint(-1, 1)
            for i in range(n - 1):
                p1 = (points[i][0] + jx, points[i][1] + jy)
                p2 = (points[i+1][0] + jx, points[i+1][1] + jy)
                cv2.line(mask, p1, p2, 255, max(1, t - off), cv2.LINE_AA)
        return
    if n == 2:
        cv2.line(mask, points[-2], points[-1], 255, t, cv2.LINE_AA)
        cv2.circle(mask, points[-1], max(1, t // 2), 255, -1, cv2.LINE_AA)
        return
    draw_quadratic_segment(mask, points[-3], points[-2], points[-1], t)

def snap_to_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 250:
        return mask
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return mask
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    new_mask = np.zeros_like(mask)
    th = max(2, stroke_base_thickness)
    if circularity > 0.82:
        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cv2.circle(new_mask, (int(cx), int(cy)), int(radius), 255, th, cv2.LINE_AA)
        return new_mask
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
    if 3 <= len(approx) <= 6:
        cv2.polylines(new_mask, [approx], True, 255, th, cv2.LINE_AA)
        return new_mask
    return mask

def analyze_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return "unknown"
    circularity = 4 * math.pi * area / (perimeter * perimeter)
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
    v = len(approx)
    if circularity > 0.85:
        return "circle"
    if v == 3:
        return "triangle"
    if v == 4:
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 1
        return "square" if 0.85 < aspect < 1.15 else "rectangle"
    if v == 5:
        return "pentagon"
    if v == 6:
        return "hexagon"
    if v > 8:
        return "complex"
    return "freehand"

# ==========================================
# ✨ UNDO STACK بهینه — فقط diff
# ==========================================
def push_undo_add():
    """ثبت اضافه شدن شیء جدید"""
    undo_stack.append({"type": "add"})
    if len(undo_stack) > MAX_UNDO_STEPS:
        undo_stack.pop(0)

def push_undo_delete(obj):
    """ثبت حذف یک شیء (باید کامل ذخیره بشه)"""
    undo_stack.append({
        "type": "delete",
        "obj": {
            "mask": obj["mask"].copy(),
            "color": obj["color"],
            "offset": list(obj["offset"]),
            "scale": obj.get("scale", 1.0),
            "target_scale": obj.get("target_scale", 1.0),
            "rotation": obj.get("rotation", 0.0),
            "bbox": list(obj.get("bbox", [0, 0, 0, 0])),
        }
    })
    if len(undo_stack) > MAX_UNDO_STEPS:
        undo_stack.pop(0)

def push_undo_clone(obj):
    """ثبت کلون شدن یک شیء"""
    undo_stack.append({"type": "add"})
    if len(undo_stack) > MAX_UNDO_STEPS:
        undo_stack.pop(0)

def perform_undo():
    """Undo بهینه بر اساس نوع آخرین عمل"""
    global objects
    if not undo_stack:
        return False

    entry = undo_stack.pop()

    if entry["type"] == "add":
        # آخرین شیء اضافه‌شده رو حذف کن
        if objects:
            obj = objects.pop()
            cx, cy = get_object_center(obj)
            spawn_particles(cx, cy, obj["color"], 8, (1.2, 3.2), (10, 18))
            spawn_ripple(cx, cy, obj["color"], 1.5)
            return True

    elif entry["type"] == "delete":
        # شیء حذف‌شده رو برگردون
        data = entry["obj"]
        new_obj = {
            "mask": data["mask"], "color": data["color"],
            "offset": list(data["offset"]), "bbox": list(data["bbox"]),
            "scale": data["scale"], "target_scale": data["target_scale"],
            "rotation": data.get("rotation", 0.0),
            "vel_x": 0.0, "vel_y": 0.0,
            "fall_vel_y": 0.0, "fall_vel_x": 0.0, "angular_velocity": 0.0, "impact_cooldown": 0, "resting": False,
        }
        objects.append(new_obj)
        cx, cy = get_object_center(new_obj)
        spawn_particles(cx, cy, new_obj["color"], 8, (1.2, 3.2), (10, 18))
        spawn_ripple(cx, cy, new_obj["color"], 1.5)
        return True

    return False

def commit_stroke():
    global current_object_mask, current_object_color, was_drawing
    global stroke_points_buffer, stroke_start_frames, stroke_current_thickness
    global drawing_hold_frames, trail_points, last_shape_name, straight_dir, straight_strength

    if current_object_mask is None:
        was_drawing = False
        stroke_start_frames = 0
        stroke_points_buffer = []
        drawing_hold_frames = 0
        trail_points = []
        straight_dir = None
        straight_strength = 0.0
        return

    if stroke_points_buffer:
        lp = stroke_points_buffer[-1]
        cv2.circle(current_object_mask, lp, max(1, int(stroke_current_thickness) // 2), 255, -1, cv2.LINE_AA)
        spawn_particles(lp[0], lp[1], current_object_color, 5, (0.7, 2.0), (7, 12))
        spawn_ripple(lp[0], lp[1], current_object_color, 0.9)

    last_shape_name = analyze_shape(current_object_mask)
    if USE_SNAP:
        current_object_mask = snap_to_shape(current_object_mask)

    # ⭐ فقط add رو push کن
    push_undo_add()

    objects.append({
        "mask": current_object_mask,
        "color": current_object_color,
        "offset": [0, 0],
        "bbox": compute_bbox(current_object_mask),
        "scale": 1.0, "target_scale": 1.0,
        "rotation": 0.0,
        "vel_x": 0.0, "vel_y": 0.0,
        "fall_vel_y": 0.0, "fall_vel_x": 0.0, "angular_velocity": 0.0, "impact_cooldown": 0, "resting": False,
    })
    current_object_mask = None
    current_object_color = None
    was_drawing = False
    stroke_points_buffer = []
    stroke_start_frames = 0
    stroke_current_thickness = stroke_base_thickness
    drawing_hold_frames = 0
    trail_points = []
    straight_dir = None
    straight_strength = 0.0

def clone_object(obj):
    global grabbed_object_id, pinch_start_dist, pinch_active, grab_initial_hand_angle, grab_initial_object_rotation, pinch_initial_angle, pinch_initial_rotation, pinch_filtered_scale, grab_angular_velocity, grab_last_rotation
    push_undo_clone(obj)
    new_obj = {
        "mask": obj["mask"].copy(),
        "color": obj["color"],
        "offset": [obj["offset"][0] + 35, obj["offset"][1] + 35],
        "bbox": list(obj.get("bbox", [0, 0, 0, 0])),
        "scale": obj.get("scale", 1.0),
        "target_scale": obj.get("target_scale", 1.0),
        "rotation": obj.get("rotation", 0.0),
        "vel_x": 0.0, "vel_y": 0.0,
        "fall_vel_y": 0.0, "fall_vel_x": 0.0, "angular_velocity": 0.0, "impact_cooldown": 0, "resting": False,
    }
    objects.append(new_obj)
    cx, cy = get_object_center(new_obj)
    spawn_particles(cx, cy, new_obj["color"], 8, (1.3, 3.2), (9, 16))
    spawn_ripple(cx, cy, new_obj["color"], 1.2)

    # ⭐ شیء جدید رو بگیر
    grabbed_object_id = len(objects) - 1
    grab_anchor_local_x = (new_obj["bbox"][0] + new_obj["bbox"][2]) // 2
    grab_anchor_local_y = (new_obj["bbox"][1] + new_obj["bbox"][3]) // 2
    grab_last_offset_x = new_obj["offset"][0]
    grab_last_offset_y = new_obj["offset"][1]
    grab_velocity_x = grab_velocity_y = 0
    grab_angular_velocity = 0.0
    grab_last_rotation = new_obj.get("rotation", 0.0)
    pinch_start_dist = None
    pinch_active = False
    grab_initial_hand_angle = None
    grab_initial_object_rotation = new_obj.get("rotation", 0.0)
    pinch_initial_angle = None
    pinch_initial_rotation = new_obj.get("rotation", 0.0)
    pinch_filtered_scale = new_obj.get("scale", 1.0)

def bring_to_front(obj_id):
    global grabbed_object_id
    if 0 <= obj_id < len(objects) - 1:  # فقط اگه در بالاترین لایه نیست
        obj = objects.pop(obj_id)
        objects.append(obj)
        if grabbed_object_id == obj_id:
            grabbed_object_id = len(objects) - 1
        elif grabbed_object_id > obj_id:
            grabbed_object_id -= 1

def apply_gravity(obj, h):
    """V12 physics: gravity, air drag, bounce, friction, and impact-aware rotation."""
    if obj.get("resting", False):
        return

    if obj.get("impact_cooldown", 0) > 0:
        obj["impact_cooldown"] -= 1

    vx = float(obj.get("fall_vel_x", 0.0))
    vy = float(obj.get("fall_vel_y", 0.0)) + GRAVITY
    vx *= AIR_DRAG
    vy *= AIR_DRAG
    vx = max(-MAX_PHYSICS_SPEED, min(MAX_PHYSICS_SPEED, vx))
    vy = max(-MAX_PHYSICS_SPEED, min(MAX_PHYSICS_SPEED, vy))

    obj["offset"][0] += vx
    obj["offset"][1] += vy
    obj["fall_vel_x"], obj["fall_vel_y"] = vx, vy

    av = float(obj.get("angular_velocity", 0.0))
    av = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, av))
    obj["rotation"] = float(obj.get("rotation", 0.0)) + av
    obj["angular_velocity"] = av * ANGULAR_DAMPING

    bbox = obj.get("bbox", [0, 0, 0, 0])
    left = obj["offset"][0] + bbox[0]
    right = obj["offset"][0] + bbox[2]
    bottom = obj["offset"][1] + bbox[3]

    if left < 8:
        impact = abs(obj["fall_vel_x"])
        obj["offset"][0] += 8 - left
        obj["fall_vel_x"] = impact * BOUNCE
        if impact > IMPACT_EFFECT_SPEED and obj.get("impact_cooldown", 0) == 0:
            spawn_particles(int(obj["offset"][0] + bbox[0]), int(obj["offset"][1] + (bbox[1] + bbox[3]) * 0.5),
                            obj.get("color", (255,255,255)), 3, (0.5, 1.5), (4, 8))
            obj["impact_cooldown"] = 4
    elif right > 1272:
        impact = abs(obj["fall_vel_x"])
        obj["offset"][0] -= right - 1272
        obj["fall_vel_x"] = -impact * BOUNCE
        if impact > IMPACT_EFFECT_SPEED and obj.get("impact_cooldown", 0) == 0:
            spawn_particles(int(obj["offset"][0] + bbox[2]), int(obj["offset"][1] + (bbox[1] + bbox[3]) * 0.5),
                            obj.get("color", (255,255,255)), 3, (0.5, 1.5), (4, 8))
            obj["impact_cooldown"] = 4

    if bottom > h - 8:
        impact = abs(float(obj["fall_vel_y"]))
        obj["offset"][1] -= bottom - (h - 8)

        # Fast impacts bounce harder; tiny contacts settle instead of jittering.
        impact_ratio = min(1.0, impact / 18.0)
        restitution = 0.22 + 0.28 * impact_ratio
        obj["fall_vel_y"] = -impact * restitution
        obj["fall_vel_x"] *= FLOOR_FRICTION
        obj["angular_velocity"] *= GROUND_ANGULAR_DAMPING

        if impact > IMPACT_EFFECT_SPEED and obj.get("impact_cooldown", 0) == 0:
            px = int(obj["offset"][0] + (bbox[0] + bbox[2]) * 0.5)
            py = max(8, int(h - 8))
            spawn_particles(px, py, obj.get("color", (255,255,255)), 5, (0.6, 1.8), (5, 10))
            spawn_ripple(px, py, obj.get("color", (255,255,255)), 0.65)
            obj["impact_cooldown"] = 5

        if abs(obj["fall_vel_y"]) < 0.75 and abs(obj.get("fall_vel_x", 0.0)) < 0.35:
            obj["fall_vel_y"] = 0.0
            obj["fall_vel_x"] = 0.0
            obj["angular_velocity"] = 0.0
            obj["resting"] = True

def object_collision_radius(obj):
    """شعاع تقریبی برای برخورد نرم بین آبجکت‌ها؛ بر اساس اندازه واقعی آبجکت و scale."""
    bbox = obj.get("bbox", [0, 0, 0, 0])
    bw = max(4.0, abs(float(bbox[2] - bbox[0])))
    bh = max(4.0, abs(float(bbox[3] - bbox[1])))
    scale = max(0.15, float(obj.get("scale", 1.0)))
    return 0.5 * math.hypot(bw, bh) * scale


def resolve_object_collisions():
    """برخورد تقریبی دایره‌ای بین آبجکت‌ها با حفظ تکانه و کمی اصطکاک."""
    count = len(objects)
    if count < 2:
        return

    for i in range(count):
        a = objects[i]
        if i == grabbed_object_id:
            continue
        ar = object_collision_radius(a)
        ax = float(a["offset"][0]) + (a.get("bbox", [0,0,0,0])[0] + a.get("bbox", [0,0,0,0])[2]) * 0.5
        ay = float(a["offset"][1]) + (a.get("bbox", [0,0,0,0])[1] + a.get("bbox", [0,0,0,0])[3]) * 0.5

        for j in range(i + 1, count):
            if j == grabbed_object_id:
                continue
            b = objects[j]
            br = object_collision_radius(b)
            bb = b.get("bbox", [0,0,0,0])
            bx = float(b["offset"][0]) + (bb[0] + bb[2]) * 0.5
            by = float(b["offset"][1]) + (bb[1] + bb[3]) * 0.5

            dx = bx - ax
            dy = by - ay
            dist_sq = dx * dx + dy * dy
            min_dist = ar + br
            if dist_sq >= min_dist * min_dist:
                continue

            if dist_sq < 1e-6:
                nx, ny = 1.0, 0.0
                dist = 0.001
            else:
                dist = math.sqrt(dist_sq)
                nx, ny = dx / dist, dy / dist

            overlap = min_dist - dist
            # اصلاح موقعیت: دو آبجکت کمی از هم جدا می‌شوند تا داخل هم نروند.
            total_mass = ar + br
            wa = br / total_mass
            wb = ar / total_mass
            a["offset"][0] -= nx * overlap * wa * COLLISION_PUSH
            a["offset"][1] -= ny * overlap * wa * COLLISION_PUSH
            b["offset"][0] += nx * overlap * wb * COLLISION_PUSH
            b["offset"][1] += ny * overlap * wb * COLLISION_PUSH

            avx = float(a.get("fall_vel_x", 0.0))
            avy = float(a.get("fall_vel_y", 0.0))
            bvx = float(b.get("fall_vel_x", 0.0))
            bvy = float(b.get("fall_vel_y", 0.0))
            rvx = bvx - avx
            rvy = bvy - avy
            rel_normal = rvx * nx + rvy * ny

            # V11: mass-aware impulse + speed-dependent restitution + friction + torque.
            if rel_normal < 0.0:
                impact_speed = -rel_normal
                restitution = COLLISION_RESTITUTION_MIN + (
                    COLLISION_RESTITUTION_MAX - COLLISION_RESTITUTION_MIN
                ) * min(1.0, impact_speed / 16.0)

                # Larger objects behave heavier; smaller ones react more strongly.
                mass_a = max(0.45, (ar / 42.0) ** 2)
                mass_b = max(0.45, (br / 42.0) ** 2)
                inv_a = 1.0 / mass_a
                inv_b = 1.0 / mass_b

                impulse = -(1.0 + restitution) * rel_normal / (inv_a + inv_b)
                avx -= impulse * nx * inv_a
                avy -= impulse * ny * inv_a
                bvx += impulse * nx * inv_b
                bvy += impulse * ny * inv_b

                tx, ty = -ny, nx
                rel_tangent = rvx * tx + rvy * ty

                # Coulomb-like friction, limited so it cannot create energy.
                jt = -rel_tangent / (inv_a + inv_b)
                max_jt = abs(impulse) * COLLISION_FRICTION
                jt = max(-max_jt, min(max_jt, jt))
                avx -= jt * tx * inv_a
                avy -= jt * ty * inv_a
                bvx += jt * tx * inv_b
                bvy += jt * ty * inv_b

                a["fall_vel_x"], a["fall_vel_y"] = avx, avy
                b["fall_vel_x"], b["fall_vel_y"] = bvx, bvy
                a["resting"] = False
                b["resting"] = False

                # Tangential impulse produces believable spin.
                spin_a = (jt / max(ar, 8.0)) * COLLISION_TORQUE
                spin_b = (jt / max(br, 8.0)) * COLLISION_TORQUE
                a["angular_velocity"] = float(a.get("angular_velocity", 0.0)) - spin_a
                b["angular_velocity"] = float(b.get("angular_velocity", 0.0)) + spin_b

                # Small impact feedback, rate-limited per object.
                if impact_speed > IMPACT_EFFECT_SPEED:
                    if a.get("impact_cooldown", 0) == 0:
                        px = int(ax + nx * ar)
                        py = int(ay + ny * ar)
                        spawn_particles(px, py, a.get("color", (255,255,255)), 3, (0.4, 1.4), (4, 8))
                        a["impact_cooldown"] = 4
                    if b.get("impact_cooldown", 0) == 0:
                        px = int(bx - nx * br)
                        py = int(by - ny * br)
                        spawn_particles(px, py, b.get("color", (255,255,255)), 3, (0.4, 1.4), (4, 8))
                        b["impact_cooldown"] = 4


def clamp_object_velocities():
    for obj in objects:
        obj["fall_vel_x"] = max(-MAX_PHYSICS_SPEED, min(MAX_PHYSICS_SPEED, float(obj.get("fall_vel_x", 0.0))))
        obj["fall_vel_y"] = max(-MAX_PHYSICS_SPEED, min(MAX_PHYSICS_SPEED, float(obj.get("fall_vel_y", 0.0))))
        obj["angular_velocity"] = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, float(obj.get("angular_velocity", 0.0))))

def update_object_scales():
    for obj in objects:
        obj.setdefault("impact_cooldown", 0)
        t = obj.get("target_scale", 1.0)
        s = obj.get("scale", 1.0)
        obj["scale"] = s + (t - s) * 0.40

def draw_trail(img):
    for tx, ty, life in trail_points:
        a = life / TRAIL_MAX_LIFE
        r = max(1, int(2.5 * a))
        cv2.circle(img, (int(tx), int(ty)), r, (int(220 * a), int(150 * a), int(255 * a)), -1, cv2.LINE_AA)

def draw_prediction(img, x, y, vx, vy):
    spd = math.hypot(vx, vy)
    if spd < 4:
        return
    length = min(45, spd * 0.08)
    ex = int(x + vx / spd * length)
    ey = int(y + vy / spd * length)
    cv2.line(img, (x, y), (ex, ey), (160, 160, 240), 1, cv2.LINE_AA)

def draw_color_picker(image, current_index):
    cell_h = PICKER_H // len(colors)
    for i, c in enumerate(colors):
        y0 = PICKER_Y + i * cell_h
        cv2.rectangle(image, (PICKER_X, y0), (PICKER_X + PICKER_W, y0 + cell_h), c, -1)
        if i == current_index:
            cv2.rectangle(image, (PICKER_X - 2, y0 - 2),
                          (PICKER_X + PICKER_W + 2, y0 + cell_h + 2), (255, 255, 255), 2)

def check_picker_hit(cx, cy):
    if not (PICKER_X <= cx <= PICKER_X + PICKER_W):
        return -1
    if not (PICKER_Y <= cy <= PICKER_Y + PICKER_H):
        return -1
    cell_h = PICKER_H // len(colors)
    idx = (cy - PICKER_Y) // cell_h
    return int(idx) if 0 <= idx < len(colors) else -1

def draw_background(image, t):
    hh, ww = image.shape[:2]
    grid = 60
    off = int(t * 10) % grid
    for x in range(-off, ww, grid):
        cv2.line(image, (x, 0), (x, hh), (16, 16, 22), 1)
    for y in range(-off, hh, grid):
        cv2.line(image, (0, y), (ww, y), (16, 16, 22), 1)

# ==========================================
# ✨ SAVE/LOAD با rotation
# ==========================================
def save_sketch(filename="sketch.npz"):
    data = {"count": len(objects)}
    for i, obj in enumerate(objects):
        data[f"mask_{i}"] = obj["mask"]
        data[f"color_{i}"] = np.array(obj["color"])
        data[f"offset_{i}"] = np.array(obj["offset"])
        data[f"scale_{i}"] = np.array([obj.get("scale", 1.0)])
        # ⭐ rotation رو هم ذخیره کن
        data[f"rotation_{i}"] = np.array([obj.get("rotation", 0.0)])
    np.savez_compressed(filename, **data)
    print(f"[Saved] {len(objects)} → {filename}")

def load_sketch(filename="sketch.npz"):
    global objects
    if not os.path.exists(filename):
        print("[Load] not found")
        return
    data = np.load(filename, allow_pickle=True)
    objects = []
    for i in range(int(data["count"])):
        m = data[f"mask_{i}"]
        # ⭐ rotation رو هم لود کن (با fallback برای فایل‌های قدیمی)
        try:
            rot = float(data[f"rotation_{i}"][0])
        except (KeyError, IndexError):
            rot = 0.0
        objects.append({
            "mask": m,
            "color": tuple(int(c) for c in data[f"color_{i}"]),
            "offset": [int(x) for x in data[f"offset_{i}"]],
            "scale": float(data[f"scale_{i}"][0]),
            "target_scale": float(data[f"scale_{i}"][0]),
            "rotation": rot,
            "vel_x": 0.0, "vel_y": 0.0,
            "fall_vel_y": 0.0, "fall_vel_x": 0.0, "angular_velocity": 0.0, "impact_cooldown": 0, "resting": False,
            "bbox": compute_bbox(m),
        })
    print(f"[Loaded] {len(objects)}")

def export_to_png(filename="artwork.png"):
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    for obj in objects:
        render_object(canvas, obj)
    if current_object_mask is not None:
        canvas[current_object_mask > 0] = current_object_color
    g1 = cv2.GaussianBlur(canvas, (5, 5), 0)
    g2 = cv2.GaussianBlur(canvas, (11, 11), 0)
    final = cv2.addWeighted(canvas, 1.0, g1, 0.32, 0)
    final = cv2.addWeighted(final, 1.0, g2, 0.15, 0)
    cv2.imwrite(filename, final)
    print(f"[Export] → {filename}")

# ==========================================
# 7. Window
# ==========================================
WINDOW_NAME = "AR Smart Board"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
try:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
except Exception:
    pass

# ==========================================
# 8. Main Loop
# ==========================================
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    output = np.zeros((h, w, 3), dtype=np.uint8)
    draw_background(output, time.perf_counter() * 0.08)

    current_mode = stable_mode
    hand_detected = False
    cx = cy = w // 2
    pulse = 0.5 + 0.5 * math.sin(time.perf_counter() * 6.0)
    mode_pulse_strength *= 0.88
    lm = None

    if results.multi_hand_landmarks:
        hand_detected = True
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            lm = hand_landmarks.landmark
            handedness = "Right"
            if results.multi_handedness:
                handedness = results.multi_handedness[idx].classification[0].label

            fingers, thumb_open = get_finger_states(lm, handedness)
            index_up, middle_up, ring_up, pinky_up = fingers
            total_fingers = sum(fingers) + (1 if thumb_open else 0)

            pinch_dist = math.hypot((lm[4].x - lm[8].x) * w, (lm[4].y - lm[8].y) * h)
            # Hysteresis: a pinch starts only when clearly closed and remains
            # active until the fingers separate more.
            if pinch_active:
                is_pinching = pinch_dist < PINCH_RELEASE
            else:
                is_pinching = pinch_dist < PINCH_START

            raw_x, raw_y = int(lm[8].x * w), int(lm[8].y * h)
            now = time.perf_counter()

            if prev_raw_x is not None and prev_t is not None:
                dt = now - prev_t
                if dt > 0:
                    inst = math.hypot(raw_x - prev_raw_x, raw_y - prev_raw_y) / dt
                    shared_speed += (inst - shared_speed) * 0.55
                    pred_vx = (raw_x - prev_raw_x) * 0.65 + pred_vx * 0.35
                    pred_vy = (raw_y - prev_raw_y) * 0.65 + pred_vy * 0.35

            prev_raw_x, prev_raw_y, prev_t = raw_x, raw_y, now

            fx = filter_x(raw_x, t=now, shared_speed=shared_speed)
            fy = filter_y(raw_y, t=now, shared_speed=shared_speed)
            cx, cy = int(round(fx)), int(round(fy))

            grab_raw_x = (cx + int(lm[4].x * w)) // 2
            grab_raw_y = (cy + int(lm[4].y * h)) // 2
            grab_x = int(filter_grab_x(grab_raw_x, t=now, shared_speed=shared_speed))
            grab_y = int(filter_grab_y(grab_raw_y, t=now, shared_speed=shared_speed))

            # --- Gestures ---
            if not any([index_up, middle_up, ring_up, pinky_up, thumb_open]):
                raw_mode = "UNDO"
            elif total_fingers >= 5:
                raw_mode = "CANCEL"
            elif index_up and thumb_open and not middle_up and not ring_up and not pinky_up:
                raw_mode = "GRAB"
            elif index_up and not middle_up and not ring_up and not pinky_up and not thumb_open:
                raw_mode = "DRAWING"
            else:
                raw_mode = "NONE"

            if grabbed_object_id >= 0 and raw_mode not in ("CANCEL", "UNDO"):
                raw_mode = "GRAB"

            if stable_mode == "DRAWING" and raw_mode != "DRAWING":
                drawing_hold_frames += 1
                if drawing_hold_frames < DRAWING_HOLD_MIN:
                    raw_mode = "DRAWING"
            elif raw_mode == "DRAWING":
                drawing_hold_frames = 0

            thresh = {"DRAWING": 1, "GRAB": 1, "UNDO": 3, "CANCEL": 2}.get(raw_mode, STABLE_THRESHOLD)
            if not mode_history or raw_mode == mode_history[-1]:
                mode_stable_count += 1
            else:
                mode_stable_count = 1
            mode_history.append(raw_mode)
            if len(mode_history) > MODE_HISTORY_SIZE:
                mode_history.pop(0)
            if mode_stable_count >= thresh:
                stable_mode = raw_mode
            current_mode = stable_mode

            if current_mode != last_stable_mode:
                mode_pulse_strength = 1.0
                spawn_ripple(cx, cy, MODE_HAND_COLORS.get(current_mode, (200, 200, 255)), 1.0)
                last_stable_mode = current_mode
                # ⭐ reset latch هنگام تغییر حالت
                bring_front_latched = False

            target_col = np.array(MODE_HAND_COLORS.get(current_mode, (0, 200, 255)), dtype=float)
            hand_color_current += (target_col - hand_color_current) * 0.22
            neon = tuple(int(c) for c in hand_color_current)

            # color picker
            if current_mode == "NONE" and not was_drawing:
                pidx = check_picker_hit(cx, cy)
                if pidx >= 0:
                    color_index = pidx

            # nearby for grab assist
            if current_mode in ("GRAB", "NONE") and grabbed_object_id < 0:
                update_nearby_and_magnet(cx, cy, 68)
                if magnet_target_id >= 0 and current_mode == "NONE":
                    apply_magnet(magnet_target_id, cx, cy, 0.10)
            else:
                nearby_object_ids.clear()
                magnet_target_id = -1

            # ===== Modes =====
            # An undo/load/reorder can invalidate the previous object index.
            # Clear it before ANY object indexing.
            validate_grab_state()

            if current_mode == "CANCEL":
                if was_drawing:
                    commit_stroke()
                lost_drawing_frames = 0
                if 0 <= grabbed_object_id < len(objects):
                    objects[grabbed_object_id]["target_scale"] = 1.0
                    spawn_particles(grab_x, grab_y, (0, 165, 255), 5, (0.9, 2.5), (7, 12))
                    grabbed_object_id = -1
                pinch_start_dist = None
                grab_initial_hand_angle = None
                prev_x = prev_y = 0

            elif current_mode == "UNDO":
                if was_drawing:
                    commit_stroke()
                lost_drawing_frames = 0
                if not undo_latched:
                    perform_undo()
                    undo_latched = True
                if validate_grab_state():
                    objects[grabbed_object_id]["target_scale"] = 1.0
                grabbed_object_id = -1
                pinch_start_dist = None
                pinch_active = False
                pinch_initial_angle = None
                grab_initial_hand_angle = None
                prev_x = prev_y = 0

            elif current_mode == "GRAB":
                if was_drawing:
                    commit_stroke()
                lost_drawing_frames = 0

                if grabbed_object_id < 0:
                    fid, ax, ay = find_nearest_object(grab_x, grab_y, 9)
                    if fid >= 0:
                        grabbed_object_id = fid
                        grab_anchor_local_x, grab_anchor_local_y = ax, ay
                        grab_last_offset_x = objects[fid]["offset"][0]
                        grab_last_offset_y = objects[fid]["offset"][1]
                        grab_velocity_x = grab_velocity_y = 0.0
                        # No artificial pop on pickup: the object keeps its real size.
                        objects[fid]["target_scale"] = objects[fid].get("scale", 1.0)
                        objects[fid]["vel_x"] = objects[fid]["vel_y"] = 0.0
                        objects[fid]["fall_vel_x"] = objects[fid].get("fall_vel_x", 0.0)
                        objects[fid]["fall_vel_y"] = objects[fid].get("fall_vel_y", 0.0)
                        objects[fid]["angular_velocity"] = 0.0
                        objects[fid]["resting"] = False
                        spawn_particles(grab_x, grab_y, objects[fid]["color"], 6, (1.0, 2.6), (7, 13))
                        spawn_ripple(grab_x, grab_y, objects[fid]["color"], 1.0)
                        pinch_start_dist = None
                        pinch_active = False
                        pinch_initial_angle = None
                        pinch_initial_rotation = objects[fid].get("rotation", 0.0)
                        pinch_filtered_scale = objects[fid].get("scale", 1.0)
                        grab_initial_hand_angle = get_hand_angle(lm)
                        grab_initial_object_rotation = objects[fid].get("rotation", 0.0)
                        grab_last_rotation = objects[fid].get("rotation", 0.0)
                        grab_angular_velocity = 0.0

                if validate_grab_state():
                    obj = objects[grabbed_object_id]

                    # 1+2) True contact grab:
                    # keep the exact point touched by the fingers under the hand,
                    # even while the object is rotated or resized.
                    depth_factor = get_hand_distance(lm, w, h)
                    # V12: a tiny adaptive prediction keeps the contact point feeling
                    # attached to the fingertip during quick hand motion, without
                    # visibly leading the hand at normal speed.
                    speed_norm = min(1.0, shared_speed / 900.0)
                    predict = GRAB_PREDICT_MIN + (GRAB_PREDICT_MAX - GRAB_PREDICT_MIN) * speed_norm
                    follow_x = grab_x + pred_vx * predict
                    follow_y = grab_y + pred_vy * predict
                    anchor_x, anchor_y = get_transformed_anchor(
                        obj, grab_anchor_local_x, grab_anchor_local_y
                    )
                    target_ox = follow_x - anchor_x
                    target_oy = follow_y - anchor_y
                    dx = target_ox - obj["offset"][0]
                    dy = target_oy - obj["offset"][1]

                    # 7+8) Adaptive follow: low speed/depth gives stability;
                    # fast movement or a closer hand gives a tighter response.
                    speed_norm = min(1.0, shared_speed / 1000.0)
                    distance_norm = min(1.0, math.hypot(dx, dy) / 80.0)
                    follow_gain = GRAB_MIN_GAIN + (GRAB_MAX_GAIN - GRAB_MIN_GAIN) * speed_norm
                    follow_gain += 0.05 * depth_factor + GRAB_RESPONSE_BOOST * distance_norm
                    follow_gain = max(GRAB_MIN_GAIN, min(0.995, follow_gain))

                    if abs(dx) > GRAB_DEADZONE or abs(dy) > GRAB_DEADZONE:
                        # Large movements respond almost immediately; tiny movements
                        # get a very small blend to suppress camera/landmark jitter.
                        micro = max(0.0, 1.0 - distance_norm) * GRAB_MICRO_SMOOTH
                        move_x = dx * follow_gain
                        move_y = dy * follow_gain
                        move_x = move_x * (1.0 - micro) + obj.get("vel_x", 0.0) * micro
                        move_y = move_y * (1.0 - micro) + obj.get("vel_y", 0.0) * micro
                        obj["offset"][0] += move_x
                        obj["offset"][1] += move_y
                        obj["vel_x"] = move_x
                        obj["vel_y"] = move_y
                    else:
                        obj["vel_x"] *= 0.22
                        obj["vel_y"] *= 0.22

                    grab_velocity_x = obj["offset"][0] - grab_last_offset_x
                    grab_velocity_y = obj["offset"][1] - grab_last_offset_y
                    grab_last_offset_x = obj["offset"][0]
                    grab_last_offset_y = obj["offset"][1]

                    # 3) Precision rotation: during Pinch, rotate from the
                    #    thumb->index line. This is much more precise than palm angle.
                    pinch_angle = get_pinch_angle(lm)
                    if pinch_active and pinch_initial_angle is not None and pinch_angle is not None:
                        rot_delta = angle_delta_deg(pinch_angle, pinch_initial_angle)
                        if abs(rot_delta) > PINCH_ROT_DEADZONE:
                            target_rot = pinch_initial_rotation + rot_delta
                            current_rot = obj.get("rotation", 0.0)
                            step = angle_delta_deg(target_rot, current_rot) * PINCH_ROT_SMOOTH
                            obj["rotation"] = current_rot + step
                    elif not pinch_active:
                        # Fallback: palm rotation when not pinching.
                        hand_angle = get_hand_angle(lm)
                        if grab_initial_hand_angle is None and hand_angle is not None:
                            grab_initial_hand_angle = hand_angle
                            grab_initial_object_rotation = obj.get("rotation", 0.0)
                        if hand_angle is not None and grab_initial_hand_angle is not None:
                            rot_delta = angle_delta_deg(hand_angle, grab_initial_hand_angle)
                            if abs(rot_delta) > 2.0:
                                target_rot = grab_initial_object_rotation + rot_delta
                                current_rot = obj.get("rotation", 0.0)
                                obj["rotation"] = current_rot + angle_delta_deg(target_rot, current_rot) * 0.20

                    # 4) Precision pinch zoom with logarithmic ratio + smoothing.
                    if pinch_active:
                        if pinch_dist >= PINCH_RELEASE:
                            pinch_active = False
                            pinch_start_dist = None
                            pinch_initial_angle = None
                        elif pinch_start_dist is not None:
                            ratio = max(0.35, min(3.2, pinch_dist / pinch_start_dist))
                            target_scale = grab_scale_at_pinch * ratio
                            max_scale = 2.8 + 0.30 * depth_factor
                            target_scale = float(np.clip(target_scale, 0.45, max_scale))
                            pinch_filtered_scale += (target_scale - pinch_filtered_scale) * PINCH_SCALE_SMOOTH
                            obj["target_scale"] = pinch_filtered_scale
                    elif is_pinching:
                        pinch_active = True
                        pinch_start_dist = max(pinch_dist, 10.0)
                        grab_scale_at_pinch = obj.get("scale", 1.0)
                        pinch_filtered_scale = grab_scale_at_pinch
                        pinch_initial_angle = pinch_angle
                        pinch_initial_rotation = obj.get("rotation", 0.0)
                        obj["target_scale"] = grab_scale_at_pinch

                    # bring_to_front only once
                    if middle_up and not ring_up and not pinky_up and not bring_front_latched:
                        bring_to_front(grabbed_object_id)
                        bring_front_latched = True
                    elif not (middle_up and not ring_up and not pinky_up):
                        bring_front_latched = False

            elif current_mode == "DRAWING":
                if validate_grab_state():
                    objects[grabbed_object_id]["target_scale"] = 1.0
                grabbed_object_id = -1
                pinch_start_dist = None
                pinch_active = False
                pinch_initial_angle = None
                grab_initial_hand_angle = None
                lost_drawing_frames = 0
                trail_points.append([cx, cy, TRAIL_MAX_LIFE])
                if len(trail_points) > 14:
                    trail_points.pop(0)

                style = PEN_STYLES[pen_style_index]
                speed_factor = min(1.0, shared_speed / 1200.0)
                pen = get_pen_params(style, speed_factor)
                hand_dist = get_hand_distance(lm, w, h) if lm is not None else 0.5

                if not was_drawing:
                    color_index = (color_index + 1) % len(colors)
                    current_object_color = colors[color_index]
                    current_object_mask = np.zeros((h, w), np.uint8)
                    stroke_points_buffer = [(cx, cy)]
                    stroke_start_frames = 0
                    stroke_current_thickness = float(pen["thickness"])
                    cv2.circle(current_object_mask, (cx, cy), max(1, int(pen["thickness"]) // 2), 255, -1, cv2.LINE_AA)
                    spawn_particles(cx, cy, current_object_color, 3, (0.5, 1.6), (5, 10))
                    prev_x, prev_y = cx, cy
                else:
                    last = stroke_points_buffer[-1] if stroke_points_buffer else (prev_x, prev_y)

                    # ⭐ Smart smooth با پارامتر سرعت
                    sm = smart_smooth_point(stroke_points_buffer, (cx, cy), shared_speed)
                    scx, scy = sm
                    dist = math.hypot(scx - last[0], scy - last[1])
                    dead = max(0.25, 0.65 - 0.40 * min(1.0, shared_speed / 900))
                    if dist >= dead:
                        stroke_points_buffer.append((scx, scy))
                        if len(stroke_points_buffer) > BUFFER_MAX:
                            stroke_points_buffer.pop(0)

                        base_th = pen["thickness"]
                        target_th = (base_th - 0.35 * speed_factor) * (1.0 + hand_dist * 0.5)
                        stroke_current_thickness += (target_th - stroke_current_thickness) * thickness_smoothing
                        th = max(1, int(stroke_current_thickness))
                        if stroke_start_frames < STROKE_START_DURATION:
                            p = stroke_start_frames / STROKE_START_DURATION
                            ease = p * p * (3 - 2 * p)
                            th = max(1, int(th * (0.55 + 0.45 * ease)))
                            stroke_start_frames += 1

                        draw_stroke_local(current_object_mask, stroke_points_buffer, th, jitter=pen["jitter"])
                        if random.random() < 0.25:
                            spawn_smart_trail(scx, scy, shared_speed, current_object_color)
                        prev_x, prev_y = scx, scy
                was_drawing = True

            else:  # NONE
                pinch_start_dist = None
                if was_drawing:
                    lost_drawing_frames += 1
                    if lost_drawing_frames > MAX_LOST_DRAWING_FRAMES:
                        commit_stroke()
                        lost_drawing_frames = 0
                prev_x = prev_y = 0

            if current_mode != "UNDO":
                undo_latched = False

            draw_neon_hand(output, hand_landmarks, w, h, neon)

            if current_mode == "DRAWING":
                th = max(2, int(stroke_current_thickness))
                cv2.circle(output, (cx, cy), th // 2 + 2, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(output, (cx, cy), 1, (255, 255, 255), -1, cv2.LINE_AA)
                draw_prediction(output, cx, cy, pred_vx, pred_vy)
            elif current_mode == "GRAB":
                cv2.circle(output, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
                if grabbed_object_id >= 0:
                    cv2.circle(output, (grab_x, grab_y), 5, neon, 1, cv2.LINE_AA)
            elif current_mode == "UNDO":
                cv2.circle(output, (cx, cy), 6, (180, 80, 255), 2, cv2.LINE_AA)

    else:
        nearby_object_ids.clear()
        magnet_target_id = -1
        if was_drawing:
            lost_drawing_frames += 1
            if lost_drawing_frames > MAX_LOST_DRAWING_FRAMES:
                commit_stroke()
                lost_drawing_frames = 0

    # 6) Inertia: keep the actual object reference so undo/reordering cannot
    # accidentally make an old index point at another object.
    if validate_grab_state() and current_mode != "GRAB":
        grabbed_obj = objects[grabbed_object_id]
        spd = math.hypot(grab_velocity_x, grab_velocity_y)
        if spd > 2.5:
            inertia_objects.append([grabbed_obj, grab_velocity_x * 0.62, grab_velocity_y * 0.62, 16])
        # رها کردن = انتقال مستقیم سرعت دست به فیزیک
        grabbed_obj["fall_vel_x"] = max(-MAX_PHYSICS_SPEED, min(MAX_PHYSICS_SPEED, grab_velocity_x * 0.65))
        grabbed_obj["fall_vel_y"] = max(-MAX_PHYSICS_SPEED, min(MAX_PHYSICS_SPEED, grab_velocity_y * 0.65))
        grabbed_obj["angular_velocity"] = max(-MAX_ANGULAR_SPEED, min(MAX_ANGULAR_SPEED, grab_angular_velocity * 1.10))
        grabbed_obj["resting"] = False
        grabbed_obj["target_scale"] = 1.0
        grabbed_object_id = -1
        pinch_start_dist = None
        pinch_active = False
        grab_angular_velocity = 0.0
        grab_last_rotation = grabbed_obj.get("rotation", 0.0)

    new_in = []
    for obj_ref, ivx, ivy, life in inertia_objects:
        if obj_ref in objects and life > 0:
            # وقتی گرانش روشن است، سرعت پرتاب مستقیماً وارد physics می‌شود؛
            # در غیر این صورت inertia کوتاه‌مدت حرکت را حفظ می‌کند.
            if not gravity_enabled:
                obj_ref["offset"][0] += ivx
                obj_ref["offset"][1] += ivy
            ivx *= 0.84
            ivy *= 0.84
            if life > 1 and math.hypot(ivx, ivy) > 0.35:
                new_in.append([obj_ref, ivx, ivy, life - 1])
    inertia_objects = new_in

    if gravity_enabled:
        for i, obj in enumerate(objects):
            if i == grabbed_object_id:
                continue
            apply_gravity(obj, h)
        # یک پاس برخورد در هر فریم؛ سبک‌تر و کافی برای تعامل نرم.
        resolve_object_collisions()
        clamp_object_velocities()

    update_object_scales()
    trail_points = [[t[0], t[1], t[2] - 1] for t in trail_points if t[2] > 1]
    update_particles()
    update_ripples()

    canvas = np.zeros((h, w, 3), np.uint8)
    for i, obj in enumerate(objects):
        render_object(canvas, obj,
                      highlight=(i == grabbed_object_id),
                      pulse=pulse,
                      is_nearby=(i in nearby_object_ids and i != grabbed_object_id))

    if current_object_mask is not None:
        canvas[current_object_mask > 0] = current_object_color

    # V13 lightweight glow: دو blur به‌جای سه blur روی کل فریم.
    g1 = cv2.GaussianBlur(canvas, (3, 3), 0)
    g2 = cv2.GaussianBlur(canvas, (11, 11), 0)
    neon_c = cv2.addWeighted(canvas, 1.0, g1, 0.26, 0)
    neon_c = cv2.addWeighted(neon_c, 1.0, g2, 0.10, 0)
    output = cv2.bitwise_or(output, neon_c)

    if mode_pulse_strength > 0.05:
        overlay = np.zeros_like(output)
        overlay[:] = MODE_HAND_COLORS.get(current_mode, (200, 200, 255))
        cv2.addWeighted(output, 1.0, overlay, mode_pulse_strength * 0.10, 0, output)

    draw_trail(output)
    draw_particles(output)
    draw_ripples(output)
    draw_color_picker(output, color_index)

    ts = w / 1600.0
    m = int(w * 0.015)
    mcol = MODE_HAND_COLORS.get(current_mode, (255, 255, 255))
    cv2.putText(output, f"Mode: {current_mode}", (m, m + int(24 * ts)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65 * ts, mcol, max(1, int(2 * ts)))
    cv2.putText(output, f"Pen: {PEN_STYLES[pen_style_index]}", (m, m + int(48 * ts)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ts, (180, 180, 220), 1)
    cv2.putText(output, f"Objects: {len(objects)}  Undo:{len(undo_stack)}", (m, m + int(70 * ts)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ts, (150, 150, 150), 1)
    if gravity_enabled:
        cv2.putText(output, "GRAVITY ON", (m, m + int(92 * ts)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ts, (0, 200, 100), 1)
    if last_shape_name:
        cv2.putText(output, f"Shape: {last_shape_name}", (m, m + int(114 * ts)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48 * ts, (200, 180, 100), 1)
    if grabbed_object_id >= 0:
        cv2.putText(output, "Grab | Pinch=Zoom+Rotate | Palm Rotate | Middle=Front | C=Clone | V11",
                    (m, m + int(136 * ts)), cv2.FONT_HERSHEY_SIMPLEX, 0.42 * ts, (0, 255, 255), 1)

    cv2.putText(output,
                "1finger=Draw  Thumb+Index=Grab  Fist=Undo  Open=Cancel  P=Pen S/L=Save/Load E=Export G=Gravity",
                (m, h - int(14 * ts)), cv2.FONT_HERSHEY_SIMPLEX, 0.38 * ts, (65, 65, 75), 1)

    if not hand_detected:
        cv2.putText(output, "Show your hand", (w // 2 - int(100 * ts), h - int(36 * ts)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * ts, (90, 90, 90), 2)

    cv2.putText(output, "Made by Mehran", (w // 2 - int(80 * ts), int(35 * ts)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5 * ts, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow(WINDOW_NAME, output)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, ord('q')):
        break
    elif key == ord('p'):
        pen_style_index = (pen_style_index + 1) % len(PEN_STYLES)
        print(f"Pen → {PEN_STYLES[pen_style_index]}")
    elif key == ord('c') and validate_grab_state():
        clone_object(objects[grabbed_object_id])
    elif key == ord('s'):
        save_sketch()
    elif key == ord('l'):
        load_sketch()
    elif key == ord('e'):
        export_to_png()
    elif key == ord('g'):
        gravity_enabled = not gravity_enabled
        if not gravity_enabled:
            for obj in objects:
                obj["resting"] = False
                obj["fall_vel_y"] = 0
                obj["fall_vel_x"] = 0
                obj["angular_velocity"] = 0
        print(f"Gravity → {'ON' if gravity_enabled else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("Done")

# ==========================================================
#                        مشخصات پروژه
# ==========================================================
#  🎨 نام پروژه: AR Smart Board
#  👤 سازنده: مهران
#
#  📢 کانال تلگرام:
#     https://t.me/MehranME_Channel
#
#  💻 مخزن گیت‌هاب:
#     https://github.com/USERNAME/REPO
#
#  📝 توضیح:
#  تخته هوشمند واقعیت افزوده با تشخیص دست.
#  با انگشت اشاره روی هوا نقاشی بکشید،
#  آبجکت‌ها را بگیرید، بچرخانید، بزرگ/کوچک کنید
#  و با فیزیک گرانش و برخورد بازی کنید.
#
#  🎮 کنترل‌ها:
#     یک انگشت (اشاره)         = طراحی
#     شست + اشاره              = گرفتن آبجکت
#     مشت                      = Undo
#     دست باز                  = Cancel
#     P                        = تغییر قلم
#     S / L                    = ذخیره / بارگذاری
#     E                        = خروجی PNG
#     G                        = روشن/خاموش کردن گرانش
#
#  📦 نصب:
#     pip install opencv-python mediapipe numpy
#
#  ▶️ اجرا:
#     python main.py
#
#  ⭐ اگر از این پروژه خوشتان آمد، در گیت‌هاب ستاره بدهید!
# ==========================================================