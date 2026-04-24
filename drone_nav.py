"""
╔══════════════════════════════════════════════════════════════════╗
║   GPS-FREE AUTONOMOUS DRONE NAVIGATION SIMULATION              ║
║   Defense-Grade Visual Navigation for GPS-Denied Environments   ║
║                                                                  ║
║   • Electronic Warfare (GPS Jamming/Spoofing)                   ║
║   • Border Surveillance & Military Reconnaissance               ║
║   • Disaster Zone Navigation                                    ║
║   • Onboard Vision + Preloaded Map — No Internet, No GPS        ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: IMPORTS
# ═══════════════════════════════════════════════════════════════════

import sys
import os
import math
import time
import json
import heapq
import subprocess
import hashlib

try:
    import numpy as np
except ImportError:
    print("[FATAL] numpy is required: pip install numpy")
    sys.exit(1)

try:
    import pygame
except ImportError:
    print("[FATAL] pygame is required: pip install pygame")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("[FATAL] opencv-python is required: pip install opencv-python")
    sys.exit(1)

from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# SECTION 2: CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# --- Colors (Military Command Center Palette) ---
BLACK       = (0, 0, 0)
WHITE       = (255, 255, 255)
GRAY        = (128, 128, 128)
DARK_GRAY   = (40, 40, 40)
GREEN       = (0, 255, 100)
DARK_GREEN  = (0, 180, 60)
RED         = (255, 60, 60)
DARK_RED    = (180, 0, 0)
CYAN        = (0, 204, 255)
DARK_CYAN   = (0, 100, 140)
YELLOW      = (255, 200, 0)
ORANGE      = (255, 140, 0)
BLUE        = (0, 120, 255)
AMBER       = (255, 176, 0)
NIGHT_TINT  = (8, 18, 8)        # Fog-of-war dark green tint

# --- Navigation ---
DRONE_SPEED       = 2.5
ARRIVAL_THRESHOLD = 10
NOISE_AMPLITUDE   = 0.5
K_NEIGHBORS       = 4

# --- Feature Extraction ---
ORB_FEATURES   = 1000
SIFT_FEATURES  = 500
MIN_KEYPOINTS  = 10

# --- Matching ---
CONFIRM_THRESHOLD = 0.22
RETRY_THRESHOLD   = 0.15
LOWE_RATIO        = 0.75
MAX_RETRIES       = 2
MATCH_TIMEOUT     = 5.0

# --- Confidence Weights ---
W_ORB   = 0.4
W_SIFT  = 0.3
W_RANSAC = 0.2
W_EDGE  = 0.1

# --- Visual ---
REVEAL_BASE_RADIUS = 80
TRAIL_MAX_LEN      = 2000
GRID_SPACING        = 50
HUD_HEIGHT          = 80
MINIMAP_W           = 180

# --- UI Panel System ---
PANEL_ALPHA         = 180
PANEL_BG            = (10, 12, 28)
NEON_GREEN          = (0, 255, 100)
PROCESS_LOG_MAX     = 5
STATUS_BANNER_TIME  = 3.0
SMOOTH_FACTOR       = 0.08

# --- Responsive UI Scaling ---
BASE_WIDTH          = 1280
BASE_HEIGHT         = 720
MIN_WINDOW_W        = 900
MIN_WINDOW_H        = 600

# --- State Machine ---
STATE_LOADING          = "LOADING"
STATE_MODE_SELECT      = "MODE_SELECT"
STATE_NODE_MARKING     = "NODE_MARKING"
STATE_NAVIGATING       = "NAVIGATING"
STATE_MISSION_COMPLETE = "MISSION_COMPLETE"
STATE_REPLAY           = "REPLAY"
STATE_POST_REPLAY      = "POST_REPLAY"

SUB_MOVING     = "MOVING"
SUB_MATCHING   = "MATCHING"
SUB_RETRYING   = "RETRYING"
SUB_REPLANNING = "REPLANNING"
SUB_COMPLETE   = "COMPLETE"

# --- Navigation Modes ---
NAV_MODE_MANUAL = "MANUAL_WAYPOINTS"
NAV_MODE_ASTAR  = "D*_LITE_ADAPTIVE"

# --- Visual Localization (GPS-Free) ---
SEARCH_WINDOW_NORMAL  = 80        # ±px search radius in NORMAL mode
SEARCH_WINDOW_SEARCH  = 120       # ±px search radius in SEARCH mode (capped)
SEARCH_GRID_STEP      = 20        # Step between candidate positions
LOCALIZATION_INTERVAL = 3         # Run localization every N frames
TEMPORAL_WEIGHT       = 0.8       # Weight for new match vs previous position
CONFIDENCE_LOW_THRESH = 0.20      # Below this → enter SEARCH mode
CONFIDENCE_LOST_THRESH = 0.10     # Below this → enter LOST mode
LOST_TIMEOUT          = 5.0       # Seconds in LOST before replanning
SPEED_NORMAL          = 2.5       # Movement speed in NORMAL mode
SPEED_SEARCH          = 1.2       # Reduced speed in SEARCH mode
SPEED_LOST            = 0.5       # Minimal speed in LOST mode

# --- Localization Modes ---
LOC_NORMAL    = "NORMAL"
LOC_SEARCHING = "SEARCHING"
LOC_LOST      = "LOST"

# --- Hardware Acceleration ---
PROC_MODE_GPU = "GPU"
PROC_MODE_CPU = "CPU"
PROC_MODE_NPU = "NPU"

# --- Replay System ---
REPLAY_EVENT_FAIL   = "FAIL"
REPLAY_EVENT_RETRY  = "RETRY"
REPLAY_EVENT_REPLAN = "REPLAN"

# --- Descriptor Cache ---
DESC_CACHE_TTL = 5   # frames

# --- Adaptive Search Window ---
SEARCH_WINDOW_TIGHT = 80   # Reduced window when confidence is high (>0.6)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def euclidean_dist(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def clamp(value, lo, hi):
    """Clamp a value to [lo, hi]."""
    return max(lo, min(value, hi))


def confidence_color(conf):
    """Map a confidence value [0,1] to a color."""
    if conf >= 0.6:
        return GREEN
    elif conf >= 0.4:
        return YELLOW
    elif conf >= 0.25:
        return ORANGE
    return RED


def pulse(t, speed=3.0, lo=0.3, hi=1.0):
    """Oscillating value for animations."""
    return lo + (hi - lo) * (0.5 + 0.5 * math.sin(t * speed))


def lerp(a, b, t):
    """Linear interpolation from a to b by factor t."""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def try_clipboard_paste():
    """Attempt to get text from system clipboard (Windows)."""
    try:
        r = subprocess.run(
            ["powershell", "-command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=2
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return ""


# ═══════════════════════════════════════════════════════════════════
# SECTION 3B: HARDWARE ACCELERATION PIPELINE
# ═══════════════════════════════════════════════════════════════════

class ProcessingPipeline:
    """Hardware-aware processing pipeline.

    Priority: GPU (CUDA) → NPU (simulated) → CPU (fallback).
    """

    def __init__(self):
        self.gpu_available = False
        self.npu_simulated = False
        self.mode = PROC_MODE_CPU
        self.init_time = 0.0
        self._detect_hardware()

    def _detect_hardware(self):
        """Attempt GPU acceleration, fall back to CPU, optionally simulate NPU."""
        t0 = time.time()
        try:
            if hasattr(cv2, 'cuda'):
                count = cv2.cuda.getCudaEnabledDeviceCount()
                if count > 0:
                    self.gpu_available = True
                    self.mode = PROC_MODE_GPU
                    print("[HW] GPU acceleration enabled")
                    self.init_time = time.time() - t0
                    return
        except Exception:
            pass

        # GPU unavailable — try NPU simulation
        self.mode = PROC_MODE_CPU
        print("[HW] Using CPU mode")

        # Simulate NPU: reduce processing overhead via smarter scheduling
        try:
            # NPU simulation — logically faster pipeline
            self.npu_simulated = True
            self.mode = PROC_MODE_NPU
            print("[HW] NPU acceleration layer (simulated)")
        except Exception:
            pass

        self.init_time = time.time() - t0

    @property
    def display_mode(self):
        return self.mode


# ═══════════════════════════════════════════════════════════════════
# SECTION 3C: DESCRIPTOR CACHE
# ═══════════════════════════════════════════════════════════════════

class DescriptorCache:
    """Cache computed descriptors to avoid redundant feature extraction.

    Stores (keypoints, descriptors) keyed by a crop hash, with TTL-based expiry.
    """

    def __init__(self, ttl=DESC_CACHE_TTL):
        self.ttl = ttl
        self._cache = {}   # key → (frame_num, keypoints, descriptors)
        self._frame = 0

    def tick(self):
        """Advance frame counter and expire old entries."""
        self._frame += 1
        expired = [k for k, v in self._cache.items() if self._frame - v[0] > self.ttl]
        for k in expired:
            del self._cache[k]

    def get(self, key):
        """Retrieve cached (kp, des) or None if miss/expired."""
        entry = self._cache.get(key)
        if entry and self._frame - entry[0] <= self.ttl:
            return entry[1], entry[2]
        return None

    def put(self, key, kp, des):
        """Store (kp, des) with current frame timestamp."""
        self._cache[key] = (self._frame, kp, des)

    def make_key(self, crop):
        """Generate a hash key from a crop image."""
        try:
            small = cv2.resize(crop, (32, 32))
            return hashlib.md5(small.tobytes()).hexdigest()
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════
# SECTION 3D: UNCERTAINTY HEATMAP (CONFIDENCE VISUALIZATION)
# ═══════════════════════════════════════════════════════════════════

class UncertaintyHeatmap:
    """Low-resolution grid that stores averaged confidence values.

    Updates only nearby cells using exponential smoothing.
    Renders a semi-transparent overlay every N frames for performance.
    """

    CELL_SIZE = 32
    SMOOTH_OLD = 0.7
    SMOOTH_NEW = 0.3
    RENDER_INTERVAL = 5  # refresh overlay every N frames

    def __init__(self, map_w, map_h):
        self.map_w = map_w
        self.map_h = map_h
        self.cols = max(1, map_w // self.CELL_SIZE)
        self.rows = max(1, map_h // self.CELL_SIZE)
        self.grid = np.full((self.rows, self.cols), 0.5, dtype=np.float32)
        self._frame = 0
        self._cached_surface = None
        self._surface_dirty = True

    def update(self, drone_pos, confidence):
        """Update local grid cells near the drone with current confidence."""
        self._frame += 1
        cx = int(drone_pos[0]) // self.CELL_SIZE
        cy = int(drone_pos[1]) // self.CELL_SIZE

        # Update 3×3 local region only
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                gx = cx + dx
                gy = cy + dy
                if 0 <= gx < self.cols and 0 <= gy < self.rows:
                    old = self.grid[gy, gx]
                    self.grid[gy, gx] = self.SMOOTH_OLD * old + self.SMOOTH_NEW * confidence

        self._surface_dirty = True

    def render(self, screen, pygame_module):
        """Render semi-transparent heatmap overlay. Refreshes every N frames."""
        if self._frame % self.RENDER_INTERVAL != 0 and self._cached_surface is not None:
            screen.blit(self._cached_surface, (0, 0))
            return

        surf = pygame_module.Surface((self.map_w, self.map_h), pygame_module.SRCALPHA)

        for gy in range(self.rows):
            for gx in range(self.cols):
                val = self.grid[gy, gx]
                # Green > 0.6, Yellow 0.3-0.6, Red < 0.3
                if val > 0.6:
                    color = (0, 255, 100, 30)
                elif val > 0.3:
                    color = (255, 200, 0, 35)
                else:
                    color = (255, 60, 60, 40)

                rx = gx * self.CELL_SIZE
                ry = gy * self.CELL_SIZE
                pygame_module.draw.rect(surf, color,
                                        (rx, ry, self.CELL_SIZE, self.CELL_SIZE))

        self._cached_surface = surf
        self._surface_dirty = False
        screen.blit(surf, (0, 0))


# ═══════════════════════════════════════════════════════════════════
# SECTION 3E: VISION DEGRADATION MODES (ROBUSTNESS TESTING)
# ═══════════════════════════════════════════════════════════════════

class VisionDegradation:
    """Runtime toggle for vision degradation effects on the drone POV patch.

    Modes: CLEAR, NOISY, BLUR, LOW_VIS, AUTO
    AUTO mode adapts based on localization confidence.
    Effects are applied ONLY to the crop patch, not the full image.
    """

    MODES = ["CLEAR", "NOISY", "BLUR", "LOW_VIS", "AUTO"]

    def __init__(self):
        self._mode_idx = 4  # default to AUTO
        self._auto_mode = "CLEAR"
        self._auto_timer = 0.0
        self._auto_cooldown = 1.0  # seconds before mode switch

    @property
    def mode(self):
        if self.MODES[self._mode_idx] == "AUTO":
            return f"AUTO ({self._auto_mode})"
        return self.MODES[self._mode_idx]

    @property
    def effective_mode(self):
        """The actual degradation mode being applied."""
        if self.MODES[self._mode_idx] == "AUTO":
            return self._auto_mode
        return self.MODES[self._mode_idx]

    def cycle(self):
        """Cycle to next vision mode."""
        self._mode_idx = (self._mode_idx + 1) % len(self.MODES)
        return self.mode

    def update_auto(self, confidence, dt):
        """Update AUTO mode based on confidence (call every frame)."""
        if self.MODES[self._mode_idx] != "AUTO":
            return
        # Determine target mode
        if confidence > 0.6:
            target = "CLEAR"
        elif confidence > 0.3:
            target = "NOISY"
        else:
            target = "LOW_VIS"
        # Cooldown: must be stable for >= 1 second
        if target != self._auto_mode:
            self._auto_timer += dt
            if self._auto_timer >= self._auto_cooldown:
                self._auto_mode = target
                self._auto_timer = 0.0
        else:
            self._auto_timer = 0.0

    def apply(self, crop):
        """Apply degradation effect to a crop patch (in-place safe).

        Returns the degraded crop. Input crop is not modified.
        """
        if crop is None or crop.size == 0:
            return crop

        mode = self.effective_mode
        if mode == "CLEAR":
            return crop

        result = crop.copy()

        if mode == "NOISY":
            noise = np.random.normal(0, 25, result.shape).astype(np.int16)
            result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        elif mode == "BLUR":
            result = cv2.GaussianBlur(result, (7, 7), 0)

        elif mode == "LOW_VIS":
            result = np.clip(result.astype(np.float32) * 0.4, 0, 255).astype(np.uint8)

        return result


# ═══════════════════════════════════════════════════════════════════
# SECTION 3F: DECISION EXPLANATION SYSTEM (REAL-TIME LOGIC)
# ═══════════════════════════════════════════════════════════════════

class DecisionExplainer:
    """Rolling log of the last N decision explanations.

    Text-only, lightweight. Each entry has a timestamp, message, and priority color.
    """

    MAX_ENTRIES = 5

    def __init__(self):
        self.entries = []  # [(timestamp, message, color), ...]

    def add(self, message, color=(0, 255, 100)):
        """Add a decision explanation entry."""
        self.entries.append((time.time(), message, color))
        if len(self.entries) > self.MAX_ENTRIES:
            self.entries.pop(0)

    def clear(self):
        self.entries.clear()

    @property
    def latest(self):
        """Return the most recent entry text, or empty string."""
        if self.entries:
            return self.entries[-1][1]
        return ""


# ═══════════════════════════════════════════════════════════════════
# SECTION 3G: LEARNING MODE (LIGHTWEIGHT MEMORY SYSTEM)
# ═══════════════════════════════════════════════════════════════════

class LearningMemory:
    """Avoid repeating known bad matches by tracking failure regions.

    Maintains a dictionary of grid_cell_id → failure_count.
    Penalizes scores in regions with high failure counts.
    Gradually decays counts over time.
    """

    GRID_SIZE = 64
    PENALTY_PER_FAILURE = 0.1
    DECAY_INTERVAL = 100  # frames between decay passes

    def __init__(self):
        self.failed_regions = {}  # (gx, gy) → failure_count
        self._frame = 0
        self.active = True

    def record_failure(self, pos):
        """Record a match failure at the given position."""
        cell = self._cell_id(pos)
        self.failed_regions[cell] = self.failed_regions.get(cell, 0) + 1

    def get_penalty(self, pos):
        """Get the score penalty for a given position."""
        if not self.active:
            return 0.0
        cell = self._cell_id(pos)
        count = self.failed_regions.get(cell, 0)
        return self.PENALTY_PER_FAILURE * count

    def adjust_score(self, original_score, pos):
        """Return adjusted score after applying failure penalty."""
        penalty = self.get_penalty(pos)
        return max(0.0, original_score - penalty)

    def tick(self):
        """Advance frame counter and decay failure counts periodically."""
        self._frame += 1
        if self._frame % self.DECAY_INTERVAL == 0:
            to_remove = []
            for cell in self.failed_regions:
                self.failed_regions[cell] = max(0, self.failed_regions[cell] - 1)
                if self.failed_regions[cell] <= 0:
                    to_remove.append(cell)
            for cell in to_remove:
                del self.failed_regions[cell]

    @property
    def penalized_count(self):
        """Number of currently penalized regions."""
        return len(self.failed_regions)

    def _cell_id(self, pos):
        return (int(pos[0]) // self.GRID_SIZE, int(pos[1]) // self.GRID_SIZE)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3H: FAILURE PREDICTION SYSTEM (PROACTIVE WARNING)
# ═══════════════════════════════════════════════════════════════════

class FailurePrediction:
    """Predict difficult regions before failure by analyzing feature density.

    Step 1: Precompute ORB feature density map (one-time).
    Step 2: During navigation, check upcoming region for low density.
    """

    GRID_SIZE = 64
    DENSITY_THRESHOLD = 0.18
    CHECK_INTERVAL = 10  # check every N frames
    LOOKAHEAD_PX = 100   # pixels ahead in movement direction
    COOLDOWN = 2.0       # seconds between triggers
    MIN_DURATION = 1.0   # minimum warning duration

    def __init__(self, map_w, map_h):
        self.map_w = map_w
        self.map_h = map_h
        self.cols = max(1, map_w // self.GRID_SIZE)
        self.rows = max(1, map_h // self.GRID_SIZE)
        self.density_grid = np.zeros((self.rows, self.cols), dtype=np.float32)
        self._frame = 0
        self.warning_active = False
        self.warning_region = None  # (gx, gy) of warned cell
        self.warning_rect = None   # (x, y, w, h) pixel rect
        self._cooldown_timer = 0.0  # time since last warning ended
        self._active_timer = 0.0   # time current warning has been active

    def precompute(self, gray_image, orb_detector):
        """Precompute feature density for each grid cell (one-time).

        Counts ORB keypoints per cell and normalizes to [0, 1].
        """
        if orb_detector is None:
            return

        max_kp = 1  # avoid div by zero

        # Count keypoints in each cell
        for gy in range(self.rows):
            for gx in range(self.cols):
                x1 = gx * self.GRID_SIZE
                y1 = gy * self.GRID_SIZE
                x2 = min(x1 + self.GRID_SIZE, gray_image.shape[1])
                y2 = min(y1 + self.GRID_SIZE, gray_image.shape[0])

                cell_crop = gray_image[y1:y2, x1:x2]
                if cell_crop.size == 0:
                    continue

                try:
                    kps = orb_detector.detect(cell_crop, None)
                    count = len(kps) if kps else 0
                    self.density_grid[gy, gx] = float(count)
                    max_kp = max(max_kp, count)
                except Exception:
                    pass

        # Normalize to [0, 1]
        if max_kp > 0:
            self.density_grid /= max_kp

    def check_ahead(self, drone_pos, heading, confidence=1.0, dt=0.016):
        """Check the region ahead of the drone for low feature density.

        Returns True if warning should be displayed.
        Balanced trigger: (density<0.18 AND conf<0.35) OR (conf<0.20)
        With 2s cooldown and 1s minimum duration.
        """
        self._frame += 1

        # Update timers
        if self.warning_active:
            self._active_timer += dt
        else:
            self._cooldown_timer += dt

        if self._frame % self.CHECK_INTERVAL != 0:
            return self.warning_active

        # Lookahead position
        lx = drone_pos[0] + math.cos(heading) * self.LOOKAHEAD_PX
        ly = drone_pos[1] + math.sin(heading) * self.LOOKAHEAD_PX

        gx = int(lx) // self.GRID_SIZE
        gy = int(ly) // self.GRID_SIZE

        should_trigger = False
        if 0 <= gx < self.cols and 0 <= gy < self.rows:
            density = self.density_grid[gy, gx]
            # Balanced trigger condition
            if (density < self.DENSITY_THRESHOLD and confidence < 0.35) or (confidence < 0.20):
                should_trigger = True

        if should_trigger and not self.warning_active:
            # Check cooldown before activating
            if self._cooldown_timer >= self.COOLDOWN:
                self.warning_active = True
                self._active_timer = 0.0
                if 0 <= gx < self.cols and 0 <= gy < self.rows:
                    self.warning_region = (gx, gy)
                    self.warning_rect = (
                        gx * self.GRID_SIZE, gy * self.GRID_SIZE,
                        self.GRID_SIZE, self.GRID_SIZE
                    )
                return True
        elif self.warning_active:
            # Enforce minimum duration before deactivating
            if not should_trigger and self._active_timer >= self.MIN_DURATION:
                self.warning_active = False
                self.warning_region = None
                self.warning_rect = None
                self._cooldown_timer = 0.0
                return False

        return self.warning_active

    def get_density_at(self, pos):
        """Get feature density at a position."""
        gx = int(pos[0]) // self.GRID_SIZE
        gy = int(pos[1]) // self.GRID_SIZE
        if 0 <= gx < self.cols and 0 <= gy < self.rows:
            return self.density_grid[gy, gx]
        return 0.0


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: A* PATHFINDING & GRAPH
# ═══════════════════════════════════════════════════════════════════

def build_knn_graph(nodes, k=K_NEIGHBORS, blocked=None):
    """Build a k-nearest-neighbor bidirectional graph.

    Args:
        nodes: dict {node_id: (x, y)}
        k: number of nearest neighbors
        blocked: set of node_ids to exclude

    Returns:
        graph: dict {node_id: [(neighbor_id, weight), ...]}
    """
    blocked = blocked or set()
    active = {nid: pos for nid, pos in nodes.items() if nid not in blocked}
    ids = list(active.keys())
    graph = defaultdict(list)

    for nid in ids:
        dists = []
        for oid in ids:
            if oid != nid:
                d = euclidean_dist(active[nid], active[oid])
                dists.append((d, oid))
        dists.sort()
        for d, oid in dists[:k]:
            if (oid, d) not in graph[nid]:
                graph[nid].append((oid, d))
            if (nid, d) not in graph[oid]:
                graph[oid].append((nid, d))
    return graph


def a_star(graph, start, goal, nodes):
    """A* shortest-path from start to goal.

    Returns:
        List of node_ids from start to goal, or None if no path.
    """
    if start == goal:
        return [start]

    open_set = [(0.0, start)]
    came_from = {}
    g = defaultdict(lambda: float("inf"))
    g[start] = 0.0
    closed = set()

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            return path[::-1]

        if cur in closed:
            continue
        closed.add(cur)

        for nb, w in graph.get(cur, []):
            if nb in closed:
                continue
            tentative = g[cur] + w
            if tentative < g[nb]:
                came_from[nb] = cur
                g[nb] = tentative
                h = euclidean_dist(nodes[nb], nodes[goal])
                heapq.heappush(open_set, (tentative + h, nb))

    return None


# ═══════════════════════════════════════════════════════════════════
# SECTION 4B: ADAPTIVE PATH PLANNER (D* LITE INSPIRED)
# ═══════════════════════════════════════════════════════════════════

class AdaptivePathPlanner:
    """D* Lite-inspired adaptive path planner.

    Maintains node cost values and a priority structure for efficient
    dynamic replanning when nodes become blocked. Initial path is
    computed with A*; subsequent replans update the graph locally
    and recompute only the affected subgraph.

    Falls back to full A* recomputation if incremental approach fails,
    ensuring stability.
    """

    def __init__(self, nodes, k=K_NEIGHBORS):
        self.nodes = dict(nodes)
        self.k = k
        self.blocked = set()
        self.g_values = {}       # node → cost from start
        self.rhs_values = {}     # node → one-step lookahead cost
        self.graph = {}
        self.current_path = []
        self.replan_count = 0
        self.goal = None
        self.start = None
        self._affected_radius = 300  # px — nodes within this of a blocked node are re-evaluated
        self._rebuild_graph()

    def _rebuild_graph(self):
        """Full graph rebuild from scratch."""
        self.graph = build_knn_graph(self.nodes, self.k, self.blocked)
        for nid in self.nodes:
            if nid not in self.blocked:
                self.g_values[nid] = float('inf')
                self.rhs_values[nid] = float('inf')

    def compute_initial_path(self, start, goal):
        """Compute initial path using A*. Returns path list or None."""
        self.start = start
        self.goal = goal
        self._rebuild_graph()
        path = a_star(self.graph, start, goal, self.nodes)
        if path:
            self.current_path = list(path)
            # Record g_values along the path
            cost = 0.0
            self.g_values[start] = 0.0
            self.rhs_values[start] = 0.0
            for i in range(1, len(path)):
                cost += euclidean_dist(self.nodes[path[i - 1]], self.nodes[path[i]])
                self.g_values[path[i]] = cost
                self.rhs_values[path[i]] = cost
        return path

    def update_blocked(self, node_id):
        """Mark a node as blocked and update the graph locally."""
        self.blocked.add(node_id)
        # Remove all edges involving the blocked node
        if node_id in self.graph:
            del self.graph[node_id]
        for nid in list(self.graph.keys()):
            self.graph[nid] = [(nb, w) for nb, w in self.graph[nid] if nb != node_id]
        # Invalidate cost values
        if node_id in self.g_values:
            self.g_values[node_id] = float('inf')
        if node_id in self.rhs_values:
            self.rhs_values[node_id] = float('inf')

    def replan_from(self, current_node, goal=None):
        """Efficient replanning from current_node.

        1. Identify nodes affected by the block (within _affected_radius).
        2. Locally rebuild graph edges for affected nodes only.
        3. Run A* on the patched graph.
        4. If that fails, do a full graph rebuild and retry.

        Returns: new path list, or None if no path exists.
        """
        goal = goal or self.goal
        if goal is None:
            return None

        self.replan_count += 1

        # --- Step 1: Identify affected nodes ---
        affected = set()
        for bid in self.blocked:
            if bid not in self.nodes:
                continue
            for nid in self.nodes:
                if nid not in self.blocked:
                    d = euclidean_dist(self.nodes[nid], self.nodes[bid])
                    if d < self._affected_radius:
                        affected.add(nid)

        # --- Step 2: Locally rebuild edges for affected nodes ---
        if affected:
            active = {nid: pos for nid, pos in self.nodes.items() if nid not in self.blocked}
            active_ids = list(active.keys())
            for nid in affected:
                if nid not in active:
                    continue
                # Recompute neighbors
                dists = []
                for oid in active_ids:
                    if oid != nid:
                        d = euclidean_dist(active[nid], active[oid])
                        dists.append((d, oid))
                dists.sort()
                self.graph[nid] = [(oid, d) for d, oid in dists[:self.k]]
                # Also ensure bidirectionality
                for oid, d in self.graph[nid]:
                    if oid in self.graph:
                        if (nid, d) not in self.graph[oid]:
                            self.graph[oid].append((nid, d))

        # --- Step 3: Run A* on patched graph ---
        path = a_star(self.graph, current_node, goal, self.nodes)

        if path is None:
            # --- Step 4: Full fallback rebuild ---
            self._rebuild_graph()
            path = a_star(self.graph, current_node, goal, self.nodes)

        if path:
            self.current_path = list(path)
            # Update cost values
            cost = 0.0
            self.g_values[current_node] = 0.0
            for i in range(1, len(path)):
                cost += euclidean_dist(self.nodes[path[i - 1]], self.nodes[path[i]])
                self.g_values[path[i]] = cost

        return path


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════

class FeatureExtractor:
    """Pre-computes ORB, SIFT, and Canny features for each node."""

    def __init__(self, cv_image_bgr, pipeline=None):
        self.image = cv_image_bgr
        self.gray = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY)
        self.gray_eq = cv2.equalizeHist(self.gray)
        self.h, self.w = self.gray.shape[:2]
        self.crop_size = int(clamp(self.w / 10, 64, 512))
        self.pipeline = pipeline or ProcessingPipeline()

        # ORB detector
        try:
            self.orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        except Exception:
            self.orb = None

        # SIFT detector (may not be available)
        self.sift = None
        try:
            self.sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
        except AttributeError:
            try:
                self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=SIFT_FEATURES)
            except Exception:
                pass
        except Exception:
            pass

        # Descriptor cache for reuse
        self.desc_cache = DescriptorCache()

        # Storage
        self.orb_features = {}   # {nid: (keypoints, descriptors, bbox)}
        self.sift_features = {}
        self.edge_features = {}  # {nid: canny_image}
        self.low_confidence = set()

    def _crop(self, pos, scale=1.0):
        """Extract a crop around pos. Returns (crop_gray, bbox)."""
        x, y = int(pos[0]), int(pos[1])
        cs = int(self.crop_size * scale)
        half = cs // 2
        x1 = clamp(x - half, 0, self.w - 1)
        y1 = clamp(y - half, 0, self.h - 1)
        x2 = clamp(x + half, 1, self.w)
        y2 = clamp(y + half, 1, self.h)
        return self.gray_eq[y1:y2, x1:x2], (x1, y1, x2, y2)

    def orb_detect_cached(self, crop):
        """ORB detect+compute with descriptor cache."""
        if self.orb is None:
            return [], None
        key = self.desc_cache.make_key(crop)
        if key:
            cached = self.desc_cache.get(key)
            if cached:
                return cached
        try:
            kp, des = self.orb.detectAndCompute(crop, None)
            if key and kp is not None:
                self.desc_cache.put(key, kp, des)
            return kp, des
        except Exception:
            return [], None

    def sift_detect_cached(self, crop):
        """SIFT detect+compute with descriptor cache."""
        if self.sift is None:
            return [], None
        key = self.desc_cache.make_key(crop)
        cache_key = f"sift_{key}" if key else None
        if cache_key:
            cached = self.desc_cache.get(cache_key)
            if cached:
                return cached
        try:
            kp, des = self.sift.detectAndCompute(crop, None)
            if cache_key and kp is not None:
                self.desc_cache.put(cache_key, kp, des)
            return kp, des
        except Exception:
            return [], None

    def extract_node(self, nid, pos):
        """Extract features for a single node."""
        crop, bbox = self._crop(pos)
        if crop.size == 0:
            self.low_confidence.add(nid)
            return

        # ORB
        try:
            kp, des = self.orb.detectAndCompute(crop, None) if self.orb else ([], None)
            self.orb_features[nid] = (kp, des, bbox)
        except Exception:
            self.orb_features[nid] = ([], None, bbox)

        # SIFT
        try:
            kp, des = self.sift.detectAndCompute(crop, None) if self.sift else ([], None)
            self.sift_features[nid] = (kp, des, bbox)
        except Exception:
            self.sift_features[nid] = ([], None, bbox)

        # Canny edges
        try:
            self.edge_features[nid] = cv2.Canny(crop, 50, 150)
        except Exception:
            self.edge_features[nid] = None

        # Low-confidence check
        n_orb = len(self.orb_features.get(nid, ([], None, None))[0] or [])
        n_sift = len(self.sift_features.get(nid, ([], None, None))[0] or [])
        if n_orb + n_sift < MIN_KEYPOINTS:
            self.low_confidence.add(nid)

    def extract_all(self, nodes):
        """Extract features for all nodes."""
        for nid, pos in nodes.items():
            self.extract_node(nid, pos)


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: PATTERN MATCHING (OPTIMIZED HYBRID)
# ═══════════════════════════════════════════════════════════════════

class PatternMatcher:
    """Hybrid ORB/SIFT matcher with geometric validation.

    Optimization Rules:
    - Skip SIFT unless ORB is weak (< 0.3)
    - Skip multi-scale unless ORB is weak
    - Use cached descriptors where possible
    """

    def __init__(self, feature_extractor):
        self.fe = feature_extractor
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # FLANN for SIFT
        self.flann = None
        try:
            idx_params = dict(algorithm=1, trees=5)
            sch_params = dict(checks=50)
            self.flann = cv2.FlannBasedMatcher(idx_params, sch_params)
        except Exception:
            pass

        self.last_confirmed_pos = None
        self.matcher_used = "ORB"
        self.match_stage = "IDLE"
        self.ransac_inlier_pct = 0.0
        self.last_match_time_ms = 0.0   # timing for performance panel
        self._low_conf_streak = 0       # consecutive low-confidence frames

    # ---- individual matchers ----

    def _orb_score(self, crop, nid):
        try:
            _, des_ref, _ = self.fe.orb_features.get(nid, ([], None, None))
            if des_ref is None or len(des_ref) < 2:
                return 0.0
            kp, des = self.fe.orb_detect_cached(crop)
            if des is None or len(des) < 2:
                return 0.0
            matches = self.bf.knnMatch(des, des_ref, k=2)
            good = sum(1 for m in matches if len(m) == 2 and m[0].distance < LOWE_RATIO * m[1].distance)
            return good / max(len(matches), 1)
        except Exception:
            return 0.0

    def _sift_score(self, crop, nid):
        try:
            if self.fe.sift is None or self.flann is None:
                return 0.0
            _, des_ref, _ = self.fe.sift_features.get(nid, ([], None, None))
            if des_ref is None or len(des_ref) < 2:
                return 0.0
            kp, des = self.fe.sift_detect_cached(crop)
            if des is None or len(des) < 2:
                return 0.0
            matches = self.flann.knnMatch(des, des_ref, k=2)
            good = sum(1 for m in matches if len(m) == 2 and m[0].distance < LOWE_RATIO * m[1].distance)
            return good / max(len(matches), 1)
        except Exception:
            return 0.0

    def _edge_score(self, crop, nid):
        try:
            ref = self.fe.edge_features.get(nid)
            if ref is None:
                return 0.0
            cur = cv2.Canny(crop, 50, 150)
            if cur.shape != ref.shape:
                cur = cv2.resize(cur, (ref.shape[1], ref.shape[0]))
            if ref.std() == 0 or cur.std() == 0:
                return 0.0
            res = cv2.matchTemplate(cur, ref, cv2.TM_CCOEFF_NORMED)
            return float(max(0, res[0][0])) if res.size > 0 else 0.0
        except Exception:
            return 0.0

    def _ransac_score(self, crop, nid):
        try:
            kp_ref, des_ref, _ = self.fe.orb_features.get(nid, ([], None, None))
            if des_ref is None or len(des_ref) < 4:
                return 0.0
            kp_cur, des_cur = self.fe.orb_detect_cached(crop)
            if des_cur is None or len(des_cur) < 4:
                return 0.0
            matches = self.bf.knnMatch(des_cur, des_ref, k=2)
            good = [m[0] for m in matches if len(m) == 2 and m[0].distance < LOWE_RATIO * m[1].distance]
            if len(good) < 4:
                return 0.0
            src = np.float32([kp_cur[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is None:
                return 0.0
            return float(np.sum(mask)) / len(good)
        except Exception:
            return 0.0

    # ---- main match routine (with optimization rules) ----

    def match(self, drone_pos, nid, node_pos, accurate=False):
        """Match the drone's current view against a target node.

        Optimization rules applied:
        1. Avoid running SIFT unless ORB fails (< 0.3)
        2. Avoid multi-scale unless needed (low conf streak > 3)
        3. Uses cached descriptors for reuse

        Returns: (confidence, status, scores_dict)
            status: 'CONFIRMED', 'RETRY', or 'FAIL'
        """
        t0 = time.time()

        crop, _ = self.fe._crop(drone_pos)
        if crop.size == 0:
            self.last_match_time_ms = 0.0
            return 0.0, "FAIL", {}

        # 1) ORB (primary — always run)
        self.match_stage = "ORB MATCHING..."
        orb = self._orb_score(crop, nid)
        self.matcher_used = "ORB"

        # 2) Multi-scale ORB ONLY if ORB weak AND we've had a streak of low conf
        if orb < 0.25 and self._low_conf_streak >= 3:
            for s in (0.75, 0.5):
                sc, _ = self.fe._crop(drone_pos, s)
                if sc.size > 0:
                    orb = max(orb, self._orb_score(sc, nid))

        # 3) SIFT fallback ONLY if ORB fails (skip if ORB is good)
        sift = 0.0
        if orb < 0.3:
            self.match_stage = "SIFT FALLBACK ACTIVE"
            sift = self._sift_score(crop, nid)
            if sift > orb:
                self.matcher_used = "SIFT"

        # 4) Geometric validation — only if we have decent scores
        ransac = 0.0
        if max(orb, sift) > 0.2:
            self.match_stage = "RANSAC VALIDATION"
            ransac = self._ransac_score(crop, nid)
            self.ransac_inlier_pct = ransac

        # 5) Edge validation
        edge = self._edge_score(crop, nid)

        # 6) Temporal consistency penalty
        penalty = 1.0
        if self.last_confirmed_pos is not None:
            d = euclidean_dist(drone_pos, self.last_confirmed_pos)
            if d > 200:
                penalty = 0.8

        # 7) Final confidence
        conf = (W_ORB * orb + W_SIFT * sift + W_RANSAC * ransac + W_EDGE * edge) * penalty

        # Track low-confidence streak for optimization decisions
        if conf < CONFIRM_THRESHOLD:
            self._low_conf_streak += 1
        else:
            self._low_conf_streak = 0

        scores = {"orb": orb, "sift": sift, "ransac": ransac, "edge": edge}

        self.last_match_time_ms = (time.time() - t0) * 1000.0

        if conf >= CONFIRM_THRESHOLD:
            self.last_confirmed_pos = drone_pos
            return conf, "CONFIRMED", scores
        elif conf >= RETRY_THRESHOLD:
            return conf, "RETRY", scores
        else:
            return conf, "FAIL", scores


# ═══════════════════════════════════════════════════════════════════
# SECTION 6B: VISUAL LOCALIZER (GPS-FREE POSITION ESTIMATION)
# ═══════════════════════════════════════════════════════════════════

class VisualLocalizer:
    """GPS-free position estimation via visual matching against the satellite map.

    The drone does NOT know its exact position. Instead, it estimates its
    position by extracting a view patch at the true position and searching
    for the best match within a local window around the estimated position.

    Optimization:
    - Uses DescriptorCache to avoid redundant ORB/SIFT extraction
    - Adaptive search window (tighter when confidence is high)
    - Skips SIFT unless ORB < 0.3
    """

    def __init__(self, feature_extractor, vision_deg=None, learning_mem=None):
        self.fe = feature_extractor
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.flann = None
        try:
            self.flann = cv2.FlannBasedMatcher(
                dict(algorithm=1, trees=5), dict(checks=50)
            )
        except Exception:
            pass

        # Intelligence integrations
        self.vision_deg = vision_deg       # VisionDegradation instance
        self.learning_mem = learning_mem   # LearningMemory instance

        # State
        self.estimated_pos = (0.0, 0.0)
        self.confidence = 0.0
        self.loc_mode = LOC_NORMAL
        self.lost_timer = 0.0
        self.frame_counter = 0

        # Enhanced UI state — cached for POV panel display
        self.last_view_crop = None
        self.last_kp_img = None
        self.last_kp_count = 0
        self.ransac_inliers_pct = 0.0
        self.matcher_used = "ORB"
        self.search_window_size = SEARCH_WINDOW_NORMAL

        # Timing for performance panel
        self.last_localize_time_ms = 0.0

    def initialize(self, start_pos):
        """Set the initial estimated position (approximate start region)."""
        # Add a small random offset to simulate not knowing exact start
        offset_x = np.random.uniform(-15, 15)
        offset_y = np.random.uniform(-15, 15)
        self.estimated_pos = (float(start_pos[0]) + offset_x,
                              float(start_pos[1]) + offset_y)
        self.confidence = 0.8
        self.loc_mode = LOC_NORMAL
        self.lost_timer = 0.0
        self.frame_counter = 0

    def localize(self, true_pos, dt):
        """Run visual localization — estimate position from map matching.

        Args:
            true_pos: The actual drone position (used ONLY to extract the
                      current view patch — simulates onboard camera).
            dt: Delta time for lost timer.
        """
        self.frame_counter += 1
        if self.frame_counter % LOCALIZATION_INTERVAL != 0:
            return

        t0 = time.time()

        # Tick the descriptor cache (expire old entries)
        self.fe.desc_cache.tick()

        # 1. Extract the drone's CURRENT VIEW at its true position
        #    (this is what the onboard camera would see)
        view_crop, _ = self.fe._crop(true_pos)
        if view_crop.size == 0:
            return

        # 1b. Apply vision degradation (if active)
        if self.vision_deg is not None:
            view_crop = self.vision_deg.apply(view_crop)

        # 2. Determine search window based on mode AND confidence
        if self.loc_mode == LOC_NORMAL and self.confidence > 0.6:
            search_r = SEARCH_WINDOW_TIGHT   # Tight window when confident
        elif self.loc_mode != LOC_NORMAL:
            search_r = SEARCH_WINDOW_SEARCH   # Wide window when searching/lost
        else:
            search_r = SEARCH_WINDOW_NORMAL
        self.search_window_size = search_r

        # Cache view for POV display with keypoints overlay
        self.last_view_crop = view_crop.copy()
        try:
            crop_bgr = cv2.cvtColor(view_crop, cv2.COLOR_GRAY2BGR)
            if self.fe.orb:
                kps, _ = self.fe.orb_detect_cached(view_crop)
                self.last_kp_count = len(kps) if kps else 0
                if kps:
                    for kp in kps[:50]:
                        kx, ky = int(kp.pt[0]), int(kp.pt[1])
                        cv2.circle(crop_bgr, (kx, ky), 3, (0, 255, 100), 1)
                        cv2.circle(crop_bgr, (kx, ky), 1, (0, 255, 255), -1)
            self.last_kp_img = crop_bgr
        except Exception:
            self.last_kp_img = None

        # 3. Search over candidate positions around estimated position
        ex, ey = self.estimated_pos
        best_score = 0.0
        best_pos = self.estimated_pos

        step = SEARCH_GRID_STEP
        for cx in range(int(ex - search_r), int(ex + search_r) + 1, step):
            for cy in range(int(ey - search_r), int(ey + search_r) + 1, step):
                cx_c = clamp(cx, 0, self.fe.w - 1)
                cy_c = clamp(cy, 0, self.fe.h - 1)

                score = self._score_candidate(view_crop, (cx_c, cy_c))
                if score > best_score:
                    best_score = score
                    best_pos = (float(cx_c), float(cy_c))

        # 4. Temporal filtering — smooth position updates
        if best_score > 0.05:
            new_x = TEMPORAL_WEIGHT * best_pos[0] + (1.0 - TEMPORAL_WEIGHT) * self.estimated_pos[0]
            new_y = TEMPORAL_WEIGHT * best_pos[1] + (1.0 - TEMPORAL_WEIGHT) * self.estimated_pos[1]
            # Limit maximum position jump per update (reduce drift)
            jump_dx = new_x - self.estimated_pos[0]
            jump_dy = new_y - self.estimated_pos[1]
            jump_dist = math.sqrt(jump_dx * jump_dx + jump_dy * jump_dy)
            max_step = 12.0
            if jump_dist > max_step:
                jscale = max_step / jump_dist
                new_x = self.estimated_pos[0] + jump_dx * jscale
                new_y = self.estimated_pos[1] + jump_dy * jscale
            self.estimated_pos = (new_x, new_y)
            self.confidence = min(best_score * 1.5, 1.0)  # scale up for display
        else:
            self.confidence = max(self.confidence - 0.02, 0.0)  # decay

        # 5. Update localization mode
        self._update_mode(dt)

        self.last_localize_time_ms = (time.time() - t0) * 1000.0

    def _score_candidate(self, view_crop, candidate_pos):
        """Score a candidate position by comparing view_crop against map at candidate_pos.

        Optimization: Skip SIFT unless ORB < 0.3. Use cached descriptors.
        Learning Memory: Penalizes regions with past failures.
        """
        map_crop, _ = self.fe._crop(candidate_pos)
        if map_crop.size == 0:
            return 0.0

        # Resize to match if needed
        if view_crop.shape != map_crop.shape:
            try:
                map_crop = cv2.resize(map_crop, (view_crop.shape[1], view_crop.shape[0]))
            except Exception:
                return 0.0

        orb = self._orb_match(view_crop, map_crop)
        self.matcher_used = "ORB"

        # OPTIMIZATION: Only run SIFT if ORB fails
        sift = 0.0
        if orb < 0.3:
            sift = self._sift_match(view_crop, map_crop)
            if sift > orb:
                self.matcher_used = "SIFT"

        # OPTIMIZATION: Only run RANSAC if we have decent scores
        ransac = 0.0
        if max(orb, sift) > 0.15:
            ransac = self._ransac_match(view_crop, map_crop)
            self.ransac_inliers_pct = ransac

        edge = self._edge_match(view_crop, map_crop)

        score = W_ORB * orb + W_SIFT * sift + W_RANSAC * ransac + W_EDGE * edge

        # Apply learning memory penalty
        if self.learning_mem is not None:
            score = self.learning_mem.adjust_score(score, candidate_pos)

        return score

    def _orb_match(self, crop_a, crop_b):
        """ORB descriptor match score between two crops (with caching)."""
        try:
            if self.fe.orb is None:
                return 0.0
            kp_a, des_a = self.fe.orb_detect_cached(crop_a)
            kp_b, des_b = self.fe.orb_detect_cached(crop_b)
            if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
                return 0.0
            matches = self.bf.knnMatch(des_a, des_b, k=2)
            good = sum(1 for m in matches if len(m) == 2
                       and m[0].distance < LOWE_RATIO * m[1].distance)
            return good / max(len(matches), 1)
        except Exception:
            return 0.0

    def _sift_match(self, crop_a, crop_b):
        """SIFT descriptor match score between two crops (with caching)."""
        try:
            if self.fe.sift is None or self.flann is None:
                return 0.0
            kp_a, des_a = self.fe.sift_detect_cached(crop_a)
            kp_b, des_b = self.fe.sift_detect_cached(crop_b)
            if des_a is None or des_b is None or len(des_a) < 2 or len(des_b) < 2:
                return 0.0
            matches = self.flann.knnMatch(des_a, des_b, k=2)
            good = sum(1 for m in matches if len(m) == 2
                       and m[0].distance < LOWE_RATIO * m[1].distance)
            return good / max(len(matches), 1)
        except Exception:
            return 0.0

    def _ransac_match(self, crop_a, crop_b):
        """RANSAC geometric validation score between two crops."""
        try:
            if self.fe.orb is None:
                return 0.0
            kp_a, des_a = self.fe.orb_detect_cached(crop_a)
            kp_b, des_b = self.fe.orb_detect_cached(crop_b)
            if des_a is None or des_b is None or len(des_a) < 4 or len(des_b) < 4:
                return 0.0
            matches = self.bf.knnMatch(des_a, des_b, k=2)
            good = [m[0] for m in matches if len(m) == 2
                    and m[0].distance < LOWE_RATIO * m[1].distance]
            if len(good) < 4:
                return 0.0
            src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is None:
                return 0.0
            return float(np.sum(mask)) / len(good)
        except Exception:
            return 0.0

    def _edge_match(self, crop_a, crop_b):
        """Canny edge template matching score."""
        try:
            ea = cv2.Canny(crop_a, 50, 150)
            eb = cv2.Canny(crop_b, 50, 150)
            if ea.shape != eb.shape:
                eb = cv2.resize(eb, (ea.shape[1], ea.shape[0]))
            if ea.std() == 0 or eb.std() == 0:
                return 0.0
            res = cv2.matchTemplate(ea, eb, cv2.TM_CCOEFF_NORMED)
            return float(max(0, res[0][0])) if res.size > 0 else 0.0
        except Exception:
            return 0.0

    def _update_mode(self, dt):
        """Update localization mode based on confidence."""
        if self.confidence >= CONFIDENCE_LOW_THRESH:
            self.loc_mode = LOC_NORMAL
            self.lost_timer = 0.0
        elif self.confidence >= CONFIDENCE_LOST_THRESH:
            self.loc_mode = LOC_SEARCHING
            self.lost_timer += dt  # track searching duration too
        else:
            self.loc_mode = LOC_LOST
            self.lost_timer += dt

    def get_speed(self):
        """Return movement speed based on confidence (adaptive)."""
        if self.loc_mode == LOC_LOST:
            return SPEED_LOST
        # Adaptive speed based on confidence level
        if self.confidence > 0.6:
            return SPEED_NORMAL * 1.2
        elif self.confidence > 0.3:
            return SPEED_NORMAL
        else:
            return SPEED_NORMAL * 0.5

    def should_replan(self):
        """Return True if the drone has been LOST too long."""
        return self.loc_mode == LOC_LOST and self.lost_timer >= LOST_TIMEOUT

    @property
    def est_ipos(self):
        return (int(self.estimated_pos[0]), int(self.estimated_pos[1]))


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: DRONE CONTROLLER
# ═══════════════════════════════════════════════════════════════════

class DroneController:
    """Smooth autonomous movement with noise and trail recording.

    GPS-Free Mode: The drone has TWO positions:
      - true position (x, y): Hidden actual position, used for camera view extraction.
      - estimated position (est_x, est_y): What the drone THINKS its position is,
        updated by the VisualLocalizer. Used for all navigation decisions.
    """

    def __init__(self, start_pos):
        # True position (hidden — simulates actual physical location)
        self.x = float(start_pos[0])
        self.y = float(start_pos[1])

        # Estimated position (what the drone believes — from visual localization)
        self.est_x = float(start_pos[0])
        self.est_y = float(start_pos[1])

        self.vx = 0.0
        self.vy = 0.0
        self.heading = 0.0
        self.trail = []          # [(x,y), ...] — uses estimated positions
        self.conf_trail = []     # [confidence, ...]

    def move_toward_estimated(self, target, speed=DRONE_SPEED):
        """Move toward target using ESTIMATED position for direction.

        The direction is computed from (est_x, est_y) to target,
        but the actual movement is applied to the TRUE position (x, y).
        Returns nothing — arrival is checked externally via estimated pos.
        """
        dx = target[0] - self.est_x
        dy = target[1] - self.est_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1.0:
            return

        # Target approach slowdown (within 30px → gradual deceleration)
        if dist < 30.0:
            speed *= max(0.3, dist / 30.0)

        new_vx = (dx / dist) * speed + np.random.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE)
        new_vy = (dy / dist) * speed + np.random.uniform(-NOISE_AMPLITUDE, NOISE_AMPLITUDE)

        # Velocity smoothing to reduce zig-zag paths
        self.vx = 0.9 * self.vx + 0.1 * new_vx
        self.vy = 0.9 * self.vy + 0.1 * new_vy

        # Move TRUE position
        self.x += self.vx
        self.y += self.vy
        self.heading = math.atan2(dy, dx)

        # Trail records ESTIMATED position (what the drone sees)
        self.trail.append((int(self.est_x), int(self.est_y)))
        if len(self.trail) > TRAIL_MAX_LEN:
            self.trail.pop(0)
            self.conf_trail.pop(0)

    def update_estimated(self, est_pos):
        """Update the estimated position from the VisualLocalizer."""
        self.est_x = float(est_pos[0])
        self.est_y = float(est_pos[1])

    @property
    def true_pos(self):
        """True position — used ONLY for camera view extraction."""
        return (self.x, self.y)

    @property
    def true_ipos(self):
        return (int(self.x), int(self.y))

    @property
    def pos(self):
        """Returns ESTIMATED position — used for all navigation."""
        return (self.est_x, self.est_y)

    @property
    def ipos(self):
        """Returns ESTIMATED position as integers."""
        return (int(self.est_x), int(self.est_y))


class UIPanel:
    """Draggable UI panel with position persistence."""

    def __init__(self, panel_id, default_x, default_y, width, height):
        self.panel_id = panel_id
        self.x = default_x
        self.y = default_y
        self.default_x = default_x
        self.default_y = default_y
        self.width = width
        self.height = height
        self.dragging = False
        self.drag_offset = (0, 0)

    def contains(self, mx, my):
        """Check if mouse is inside panel."""
        return (self.x <= mx <= self.x + self.width and
                self.y <= my <= self.y + self.height)

    def start_drag(self, mx, my):
        self.dragging = True
        self.drag_offset = (mx - self.x, my - self.y)

    def update_drag(self, mx, my, scr_w, scr_h):
        if self.dragging:
            self.x = clamp(mx - self.drag_offset[0], 0, scr_w - self.width)
            self.y = clamp(my - self.drag_offset[1], 0, scr_h - self.height)

    def stop_drag(self):
        self.dragging = False

    def reset(self):
        self.x = self.default_x
        self.y = self.default_y

    def to_dict(self):
        return {"x": self.x, "y": self.y}

    def from_dict(self, d):
        self.x = d.get("x", self.default_x)
        self.y = d.get("y", self.default_y)


class UIPanelManager:
    """Manages all draggable panels with save/load/reset."""

    def __init__(self):
        self.panels = {}
        self._active_drag = None

    def register(self, panel):
        self.panels[panel.panel_id] = panel

    def get(self, panel_id):
        return self.panels.get(panel_id)

    def handle_mouse_down(self, mx, my):
        """Start dragging if mouse is inside a panel. Returns True if handled."""
        for pid in reversed(list(self.panels.keys())):
            p = self.panels[pid]
            if p.contains(mx, my):
                p.start_drag(mx, my)
                self._active_drag = pid
                return True
        return False

    def handle_mouse_move(self, mx, my, scr_w, scr_h):
        if self._active_drag and self._active_drag in self.panels:
            self.panels[self._active_drag].update_drag(mx, my, scr_w, scr_h)

    def handle_mouse_up(self):
        if self._active_drag and self._active_drag in self.panels:
            self.panels[self._active_drag].stop_drag()
        self._active_drag = None

    def reset_all(self):
        for p in self.panels.values():
            p.reset()

    def save(self, filepath):
        data = {}
        for pid, p in self.panels.items():
            data[pid] = p.to_dict()
        try:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception:
            return False

    def load(self, filepath):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            for pid, d in data.items():
                if pid in self.panels:
                    self.panels[pid].from_dict(d)
            return True
        except Exception:
            return False


# ═══════════════════════════════════════════════════════════════════
# SECTION 7B: REPLAY SYSTEM (ENHANCED)
# ═══════════════════════════════════════════════════════════════════

class ReplaySystem:
    """Enhanced replay system with smooth interpolation, event markers,
    timeline, speed control, pause/resume, seek, and timestamp display.
    """

    SPEEDS = [0.25, 0.5, 1.0, 2.0]

    def __init__(self):
        self.trail = []          # [(x, y), ...]
        self.confs = []          # [confidence, ...]
        self.events = {}         # {trail_index: event_type}  — FAIL, RETRY, REPLAN
        self.speed = 1.0         # default 1x speed
        self.speed_idx = 2       # index into SPEEDS (1.0x)
        self.position = 0.0     # Fractional replay position
        self.playing = True
        self.paused = False

    def record_point(self, pos, conf):
        """Record a trail point with confidence."""
        self.trail.append((int(pos[0]), int(pos[1])))
        self.confs.append(conf)

    def record_event(self, event_type):
        """Record an event at the current trail position."""
        idx = max(0, len(self.trail) - 1)
        self.events[idx] = event_type

    def reset(self):
        """Reset replay position (but keep data)."""
        self.position = 0.0
        self.playing = True
        self.paused = False

    def set_speed(self, speed):
        """Set replay speed."""
        self.speed = clamp(speed, 0.25, 3)

    def cycle_speed(self, direction=1):
        """Cycle through preset speeds."""
        self.speed_idx = (self.speed_idx + direction) % len(self.SPEEDS)
        self.speed = self.SPEEDS[self.speed_idx]

    def toggle_pause(self):
        """Toggle pause/resume."""
        self.paused = not self.paused

    def seek(self, delta_points):
        """Seek forward/backward by a number of trail points."""
        if not self.trail:
            return
        self.position = clamp(self.position + delta_points, 0.0, float(len(self.trail) - 1))

    def seek_to(self, progress):
        """Seek to a specific progress value [0, 1]."""
        if not self.trail:
            return
        self.position = progress * (len(self.trail) - 1)

    def advance(self, dt):
        """Advance replay position by dt * speed. Returns True if finished."""
        if not self.trail or not self.playing:
            return True
        if self.paused:
            return False
        # Advance by speed * base_step per frame
        base_step = 3.0
        self.position += base_step * self.speed
        if self.position >= len(self.trail) - 1:
            self.position = float(len(self.trail) - 1)
            self.playing = False
            return True
        return False

    def get_interpolated_pos(self):
        """Get smoothly interpolated position at current fractional index."""
        if not self.trail:
            return (0, 0)

        idx = int(self.position)
        frac = self.position - idx

        if idx >= len(self.trail) - 1:
            return self.trail[-1]

        # Linear interpolation between adjacent points
        p1 = self.trail[idx]
        p2 = self.trail[min(idx + 1, len(self.trail) - 1)]

        ix = lerp(p1[0], p2[0], frac)
        iy = lerp(p1[1], p2[1], frac)
        return (int(ix), int(iy))

    def get_event_at(self, idx):
        """Get event type at a trail index, or None."""
        return self.events.get(idx)

    def get_conf_at(self, idx):
        """Get confidence at trail index (clamped)."""
        if not self.confs:
            return 0.5
        i = clamp(idx, 0, len(self.confs) - 1)
        return self.confs[i]

    def get_progress(self):
        """Get replay progress [0, 1]."""
        if not self.trail:
            return 0.0
        return self.position / max(len(self.trail) - 1, 1)

    @property
    def current_idx(self):
        return int(self.position)

    @property
    def total_points(self):
        return len(self.trail)


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: UI DRAWING HELPERS
# ═══════════════════════════════════════════════════════════════════

def draw_corner_brackets(surf, rect, color, size=20, thick=2):
    """Military-style corner brackets on a rectangle."""
    x, y, w, h = rect
    segs = [
        # Top-left
        ((x, y + size), (x, y), (x + size, y)),
        # Top-right
        ((x + w - size, y), (x + w, y), (x + w, y + size)),
        # Bottom-left
        ((x, y + h - size), (x, y + h), (x + size, y + h)),
        # Bottom-right
        ((x + w - size, y + h), (x + w, y + h), (x + w, y + h - size)),
    ]
    for a, b, c in segs:
        pygame.draw.line(surf, color, a, b, thick)
        pygame.draw.line(surf, color, b, c, thick)


def draw_scanline(surf, w, h, t, color=(0, 200, 255, 25)):
    """Animated horizontal scan line."""
    sy = int(h * ((t * 0.25) % 1.0))
    s = pygame.Surface((w, 2), pygame.SRCALPHA)
    s.fill(color)
    surf.blit(s, (0, sy))


def draw_progress_bar(surf, x, y, w, h, value, fg, bg=DARK_GRAY):
    """Horizontal progress/bar indicator (value 0-1)."""
    pygame.draw.rect(surf, bg, (x, y, w, h), border_radius=3)
    fill_w = int(w * clamp(value, 0, 1))
    if fill_w > 0:
        pygame.draw.rect(surf, fg, (x, y, fill_w, h), border_radius=3)
    pygame.draw.rect(surf, GRAY, (x, y, w, h), 1, border_radius=3)


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════

def main():
    pygame.init()

    # --- Window ---
    scr_w, scr_h = 1200, 800
    screen = pygame.display.set_mode((scr_w, scr_h), pygame.RESIZABLE)
    pygame.display.set_caption("GPS-FREE DRONE NAVIGATION — DEFENSE SIMULATION")
    clock = pygame.time.Clock()

    # --- UI Scaling ---
    ui_scale = min(scr_w / BASE_WIDTH, scr_h / BASE_HEIGHT)
    ui_scale = max(ui_scale, 0.6)  # minimum scale floor

    def s(val):
        """Scale a pixel value by the UI scale factor."""
        return max(1, int(val * ui_scale))

    # --- Fonts (scaled) ---
    try:
        f_title = pygame.font.SysFont("consolas", s(34), bold=True)
        f_large = pygame.font.SysFont("consolas", s(26), bold=True)
        f_med   = pygame.font.SysFont("consolas", s(19))
        f_small = pygame.font.SysFont("consolas", s(15))
        f_hud   = pygame.font.SysFont("consolas", s(13))
    except Exception:
        f_title = pygame.font.Font(None, s(34))
        f_large = pygame.font.Font(None, s(26))
        f_med   = pygame.font.Font(None, s(19))
        f_small = pygame.font.Font(None, s(15))
        f_hud   = pygame.font.Font(None, s(13))

    # ======================== STATE ========================
    state       = STATE_LOADING
    sub_state   = ""

    # Image
    sat_rgb     = None   # numpy (H,W,3) RGB display-resolution
    sat_surface = None   # pygame surface
    cv_display  = None   # numpy (H,W,3) BGR for OpenCV (display-res)
    img_w = img_h = 0

    # Nodes
    nodes       = {}     # {id: (x,y)}
    node_types  = {}     # {id: "source"/"dest"/"normal"}
    source_id   = None
    dest_id     = None
    node_ctr    = 0

    # Graph / path
    graph       = {}
    path        = []
    path_idx    = 0
    blocked     = set()

    # Features / Matching
    feat_ext    = None
    matcher     = None

    # Drone
    drone       = None
    confidence  = 0.0
    match_scores = {}
    match_retries = 0
    match_t0    = 0
    confirmed   = []
    skipped     = []

    # Visual Localizer (GPS-Free)
    localizer   = None
    show_true_pos = False   # Toggle with T key — shows true pos debug overlay

    # Reveal
    reveal_mask = None
    reveal_surface_cache = None
    reveal_dirty = True
    last_reveal_pos = None
    dark_image  = None

    # Visual / timing
    show_grid   = False
    edge_snap   = False
    accurate_mode = False
    anim_t      = 0.0
    flash_t     = 0.0
    flash_col   = None
    replan_flash = 0.0
    status_msg  = ""
    status_col  = WHITE

    # Navigation mode
    nav_mode    = NAV_MODE_ASTAR   # default

    # Mission
    mission_t0  = 0
    mission_t1  = 0

    # Replay (ENHANCED)
    replay_system = ReplaySystem()
    replay_done = False  # True after first replay finishes

    # Loading
    load_t0     = time.time()
    input_text  = ""
    input_active = False

    # Minimap
    minimap_surf = None
    minimap_h   = 0

    # --- Enhanced UI State ---
    process_log = []           # [(timestamp, message, color), ...]
    status_banner_text = ""
    status_banner_color = CYAN
    status_banner_timer = 0.0
    smooth_confidence = 0.0
    pov_matcher_status = ""
    frame_time_ms = 0.0

    # --- Hardware Acceleration Pipeline ---
    pipeline = ProcessingPipeline()

    # --- Adaptive Path Planner ---
    planner = None

    # --- Performance Monitoring (Extended) ---
    match_time_ms = 0.0
    localize_time_ms = 0.0
    fps_history = []           # last 60 FPS values for sparkline

    # --- Advanced Intelligence Systems ---
    heatmap           = None   # UncertaintyHeatmap instance
    vision_degradation = VisionDegradation()  # Vision mode (always available)
    decision_explainer = DecisionExplainer()  # Decision explanation log
    learning_memory   = LearningMemory()      # Failure memory system
    failure_predictor = None   # FailurePrediction instance (needs map dims)

    # --- Elite Demo Features ---
    smooth_heading        = 0.0               # Smoothed heading for compass
    confidence_history    = []                # Last 50 confidence values for graph
    conf_hist_frame       = 0                 # Frame counter for history sampling

    # Target lock animation
    target_lock_active    = False
    target_lock_pos       = (0, 0)
    target_lock_timer     = 0.0
    target_lock_color     = GREEN
    target_lock_duration  = 0.5

    # Camera follow (smooth offset)
    cam_x = 0.0
    cam_y = 0.0
    cam_active = False       # Activates when map is larger than screen

    # --- Draggable Panel System ---
    panel_mgr = UIPanelManager()
    layout_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "ui_layout.json")

    # Register panels with default positions
    panel_mgr.register(UIPanel("pov", scr_w - int(scr_w * 0.25) - int(scr_w * 0.02), int(scr_h * 0.02), int(scr_w * 0.25), int(scr_w * 0.25 * 0.75) + s(55)))
    panel_mgr.register(UIPanel("performance", int(scr_w * 0.02), scr_h - 260, s(260), s(130)))
    panel_mgr.register(UIPanel("decision", int(scr_w * 0.02), scr_h - HUD_HEIGHT - s(200), s(260), s(160)))
    panel_mgr.register(UIPanel("confidence", int(scr_w * 0.02), int(scr_h * 0.35), s(170), s(80)))
    panel_mgr.register(UIPanel("process_log", int(scr_w * 0.02), scr_h - HUD_HEIGHT - s(120), s(260), s(100)))

    # Load saved layout (if exists)
    panel_mgr.load(layout_path)

    # --- Fullscreen ---
    is_fullscreen = False

    # ======================== HELPERS ========================

    def set_status(msg, col=WHITE):
        nonlocal status_msg, status_col
        status_msg = msg
        status_col = col

    def add_log(msg, col=NEON_GREEN):
        """Add a message to the process flow log."""
        process_log.append((time.time(), msg, col))
        if len(process_log) > PROCESS_LOG_MAX:
            process_log.pop(0)

    def set_banner(msg, col=CYAN):
        """Set the top-center status banner text."""
        nonlocal status_banner_text, status_banner_color, status_banner_timer
        status_banner_text = msg
        status_banner_color = col
        status_banner_timer = STATUS_BANNER_TIME

    def load_image(filepath):
        nonlocal sat_rgb, sat_surface, cv_display, img_w, img_h
        nonlocal screen, scr_w, scr_h, dark_image, reveal_mask
        nonlocal minimap_surf, minimap_h, reveal_dirty

        try:
            raw = cv2.imread(filepath)
            if raw is None:
                return False, "Failed to read image file"
        except Exception as e:
            return False, f"OpenCV error: {e}"

        oh, ow = raw.shape[:2]
        max_w, max_h = 1600, 900
        scale = min(max_w / ow, max_h / oh, 1.0)
        dw, dh = int(ow * scale), int(oh * scale)

        disp = cv2.resize(raw, (dw, dh), interpolation=cv2.INTER_AREA)
        cv_display = disp
        sat_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        sat_surface = pygame.surfarray.make_surface(sat_rgb.swapaxes(0, 1))
        img_w, img_h = dw, dh
        scr_w, scr_h = dw, dh + HUD_HEIGHT
        screen = pygame.display.set_mode((scr_w, scr_h))

        # Dark version for fog-of-war
        dark_image = np.full_like(sat_rgb, NIGHT_TINT, dtype=np.uint8)
        reveal_mask = np.zeros((dh, dw), dtype=np.uint8)
        reveal_dirty = True

        # Minimap
        mm_w = MINIMAP_W
        mm_h = int(MINIMAP_W * dh / dw)
        minimap_h = mm_h
        minimap_surf = pygame.transform.smoothscale(sat_surface, (mm_w, mm_h))

        return True, f"Loaded {ow}x{oh} → display {dw}x{dh}"

    def compute_reveal():
        """Recompute the fog-of-war blended surface (cached)."""
        nonlocal reveal_surface_cache, reveal_dirty
        if not reveal_dirty:
            return
        mask3 = np.broadcast_to(reveal_mask[:, :, np.newaxis] > 0, sat_rgb.shape)
        blended = np.where(mask3, sat_rgb, dark_image)
        reveal_surface_cache = pygame.surfarray.make_surface(blended.swapaxes(0, 1))
        reveal_dirty = False

    def update_reveal():
        """Stamp the reveal mask at the drone's TRUE position (camera sees reality)."""
        nonlocal last_reveal_pos, reveal_dirty
        if drone is None:
            return
        r = int(REVEAL_BASE_RADIUS * (img_w / 800))
        r = max(r, 40)
        # Use true position for reveal — the camera sees from the actual location
        true_ip = drone.true_ipos
        if last_reveal_pos is not None:
            cv2.line(reveal_mask, last_reveal_pos, true_ip, 255, r * 2)
        cv2.circle(reveal_mask, true_ip, r, 255, -1)
        if last_reveal_pos != true_ip:
            reveal_dirty = True
        last_reveal_pos = true_ip

    def snap_to_edge(pos, radius=20):
        """Snap position to nearest Canny edge point."""
        if cv_display is None:
            return pos
        try:
            gray = cv2.cvtColor(cv_display, cv2.COLOR_BGR2GRAY)
            x, y = int(pos[0]), int(pos[1])
            y1 = max(0, y - radius)
            y2 = min(gray.shape[0], y + radius)
            x1 = max(0, x - radius)
            x2 = min(gray.shape[1], x + radius)
            crop = gray[y1:y2, x1:x2]
            edges = cv2.Canny(crop, 50, 150)
            pts = np.argwhere(edges > 0)
            if len(pts) == 0:
                return pos
            best_d = float("inf")
            best_p = pos
            for ey, ex in pts:
                ax, ay = x1 + ex, y1 + ey
                d = euclidean_dist((ax, ay), pos)
                if d < best_d:
                    best_d = d
                    best_p = (ax, ay)
            return best_p
        except Exception:
            return pos

    def start_navigation():
        nonlocal state, sub_state, graph, path, path_idx
        nonlocal feat_ext, matcher, drone, mission_t0
        nonlocal reveal_mask, reveal_dirty, last_reveal_pos
        nonlocal localizer, planner
        nonlocal heatmap, failure_predictor

        node_pos = {nid: nodes[nid] for nid in nodes}

        if nav_mode == NAV_MODE_MANUAL:
            # ---- Manual waypoint mode ----
            # Drone follows nodes in the exact order placed by the user:
            # source → normal nodes in placement order → destination
            ordered_ids = sorted(nodes.keys())  # placement order = id order
            # Build path: source first, then intermediates, then dest last
            p = [source_id]
            for nid in ordered_ids:
                if nid != source_id and nid != dest_id:
                    p.append(nid)
            p.append(dest_id)
        else:
            # ---- A* shortest path mode (via AdaptivePathPlanner) ----
            planner = AdaptivePathPlanner(nodes, K_NEIGHBORS)
            p = planner.compute_initial_path(source_id, dest_id)
            if p is None:
                set_status("NO VALID PATH — Check node placement", RED)
                return False
            graph.update(planner.graph)

        # Feature extraction (with hardware pipeline)
        feat_ext = FeatureExtractor(cv_display, pipeline)
        feat_ext.extract_all(nodes)
        matcher = PatternMatcher(feat_ext)

        path[:] = p
        path_idx = 1  # index 0 is source, we start moving to 1

        drone_obj = DroneController(nodes[source_id])
        nonlocal confidence
        confidence = 1.0

        # Initialize Visual Localizer (GPS-Free position estimation)
        # Pass vision degradation and learning memory for integration
        localizer = VisualLocalizer(feat_ext, vision_degradation, learning_memory)
        localizer.initialize(nodes[source_id])
        # Sync drone's estimated position with localizer initial estimate
        drone_obj.update_estimated(localizer.estimated_pos)

        # --- Initialize Advanced Intelligence Systems ---

        # Uncertainty Heatmap
        heatmap = UncertaintyHeatmap(img_w, img_h)

        # Failure Prediction (precompute feature density map)
        failure_predictor = FailurePrediction(img_w, img_h)
        failure_predictor.precompute(feat_ext.gray, feat_ext.orb)

        # Reset decision explainer & learning memory for new mission
        decision_explainer.clear()
        learning_memory.failed_regions.clear()
        learning_memory._frame = 0

        decision_explainer.add("Mission started — systems online", CYAN)
        decision_explainer.add(f"Navigation mode: {nav_mode}", WHITE)

        # Reset reveal
        reveal_mask[:] = 0
        reveal_dirty = True
        last_reveal_pos = None

        confirmed.clear()
        skipped.clear()
        confirmed.append(source_id)

        # Reset replay system
        replay_system.trail.clear()
        replay_system.confs.clear()
        replay_system.events.clear()
        replay_system.position = 0.0

        mission_t0 = time.time()
        state = STATE_NAVIGATING
        sub_state = SUB_MOVING

        set_banner(f"ADAPTIVE REPLANNING ACTIVE", GREEN)
        add_log(f"Processing Mode: {pipeline.display_mode}", CYAN)

        # Store drone in outer scope via nonlocal
        return drone_obj

    # ======================== DRAW: MODE SELECT ========================

    def draw_mode_select():
        screen.fill((5, 5, 20))
        t = time.time()
        glow = int(180 + 75 * math.sin(t * 2))

        # Border
        br = pygame.Rect(30, 20, scr_w - 60, scr_h - 40)
        pygame.draw.rect(screen, (0, 60, 120), br, 1, border_radius=8)
        draw_corner_brackets(screen, (30, 20, scr_w - 60, scr_h - 40), CYAN, 25, 2)
        draw_scanline(screen, scr_w, scr_h, t)

        # Title
        t1 = f_title.render("SELECT NAVIGATION MODE", True, (0, glow, 255))
        screen.blit(t1, (scr_w // 2 - t1.get_width() // 2, 55))
        t2 = f_med.render("Choose how the drone will navigate to destination", True, DARK_CYAN)
        screen.blit(t2, (scr_w // 2 - t2.get_width() // 2, 100))

        # Divider
        pygame.draw.line(screen, DARK_CYAN, (80, 140), (scr_w - 80, 140), 1)

        # Option 1 box
        opt1_selected = nav_mode == NAV_MODE_MANUAL
        opt1_y = 170
        opt1_rect = pygame.Rect(60, opt1_y, scr_w - 120, 130)
        opt1_border = GREEN if opt1_selected else GRAY
        panel1 = pygame.Surface((opt1_rect.w, opt1_rect.h), pygame.SRCALPHA)
        panel1.fill((10, 30, 10, 180) if opt1_selected else (15, 15, 40, 140))
        screen.blit(panel1, opt1_rect.topleft)
        pygame.draw.rect(screen, opt1_border, opt1_rect, 2, border_radius=6)

        key1 = f_large.render("[1]", True, GREEN if opt1_selected else GRAY)
        screen.blit(key1, (opt1_rect.x + 20, opt1_rect.y + 15))
        title1 = f_large.render("MANUAL WAYPOINTS", True, WHITE)
        screen.blit(title1, (opt1_rect.x + 80, opt1_rect.y + 15))
        desc1_lines = [
            "Drone follows nodes in the exact order you place them.",
            "Path: Source → Node 1 → Node 2 → ... → Destination",
            "Best for: Precise route control, specific surveillance paths",
        ]
        for i, dl in enumerate(desc1_lines):
            screen.blit(f_small.render(dl, True, DARK_GREEN if opt1_selected else DARK_GRAY),
                        (opt1_rect.x + 80, opt1_rect.y + 52 + i * 22))

        # Option 2 box
        opt2_selected = nav_mode == NAV_MODE_ASTAR
        opt2_y = 320
        opt2_rect = pygame.Rect(60, opt2_y, scr_w - 120, 130)
        opt2_border = GREEN if opt2_selected else GRAY
        panel2 = pygame.Surface((opt2_rect.w, opt2_rect.h), pygame.SRCALPHA)
        panel2.fill((10, 30, 10, 180) if opt2_selected else (15, 15, 40, 140))
        screen.blit(panel2, opt2_rect.topleft)
        pygame.draw.rect(screen, opt2_border, opt2_rect, 2, border_radius=6)

        key2 = f_large.render("[2]", True, GREEN if opt2_selected else GRAY)
        screen.blit(key2, (opt2_rect.x + 20, opt2_rect.y + 15))
        title2 = f_large.render("D* LITE ADAPTIVE", True, WHITE)
        screen.blit(title2, (opt2_rect.x + 80, opt2_rect.y + 15))
        desc2_lines = [
            "Drone uses D* Lite adaptive pathfinding algorithm.",
            "Path: K-NN graph + incremental replanning on failure",
            "Best for: Dynamic environments, GPS-denied operations",
        ]
        for i, dl in enumerate(desc2_lines):
            screen.blit(f_small.render(dl, True, DARK_GREEN if opt2_selected else DARK_GRAY),
                        (opt2_rect.x + 80, opt2_rect.y + 52 + i * 22))

        # Selected indicator
        sel_name = "MANUAL WAYPOINTS" if nav_mode == NAV_MODE_MANUAL else "D* LITE ADAPTIVE"
        sel_txt = f_med.render(f"Selected: {sel_name}", True, GREEN)
        screen.blit(sel_txt, (scr_w // 2 - sel_txt.get_width() // 2, 480))

        # Bottom hint
        hint = "Press 1 or 2 to select  |  ENTER to confirm  |  ESC to quit"
        screen.blit(f_hud.render(hint, True, GRAY), (scr_w // 2 - f_hud.size(hint)[0] // 2, scr_h - 30))

    # ======================== DRAW: POST REPLAY ========================

    def draw_post_replay():
        screen.fill((5, 5, 18))
        if sat_surface:
            ds = sat_surface.copy()
            ds.set_alpha(30)
            screen.blit(ds, (0, 0))

        t = time.time()
        glow = int(180 + 75 * math.sin(t * 3))

        pw, ph = 420, 200
        px = scr_w // 2 - pw // 2
        py = scr_h // 2 - ph // 2

        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((8, 8, 35, 235))
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, (0, glow, 255), (px, py, pw, ph), 2, border_radius=8)
        draw_corner_brackets(screen, (px, py, pw, ph), CYAN, 15, 2)

        title = f_large.render("REPLAY COMPLETE", True, CYAN)
        screen.blit(title, (scr_w // 2 - title.get_width() // 2, py + 25))

        pygame.draw.line(screen, DARK_CYAN, (px + 30, py + 60), (px + pw - 30, py + 60), 1)

        # Options
        opt_r = f_med.render("[R]  Replay Again", True, GREEN)
        opt_q = f_med.render("[ESC]  Exit Simulation", True, RED)
        screen.blit(opt_r, (scr_w // 2 - opt_r.get_width() // 2, py + 85))
        screen.blit(opt_q, (scr_w // 2 - opt_q.get_width() // 2, py + 125))

    # ======================== DRAW: LOADING ========================

    def draw_loading():
        screen.fill((5, 5, 20))
        t = time.time()
        glow = int(180 + 75 * math.sin(t * 2))

        # Border
        br = pygame.Rect(30, 20, scr_w - 60, scr_h - 40)
        pygame.draw.rect(screen, (0, 60, 120), br, 1, border_radius=8)
        draw_corner_brackets(screen, (30, 20, scr_w - 60, scr_h - 40), CYAN, 25, 2)
        draw_scanline(screen, scr_w, scr_h, t)

        # Title
        t1 = f_title.render("GPS-FREE DRONE NAVIGATION", True, (0, glow, 255))
        screen.blit(t1, (scr_w // 2 - t1.get_width() // 2, 55))
        t2 = f_med.render("DEFENSE SIMULATION SYSTEM v3.0", True, DARK_CYAN)
        screen.blit(t2, (scr_w // 2 - t2.get_width() // 2, 95))

        # Context lines
        lines = [
            "▸ GPS-Denied Environment Simulation",
            "▸ Electronic Warfare Countermeasure",
            "▸ Visual Odometry + Feature Matching",
            "▸ D* Lite-Inspired Adaptive Replanning",
            "▸ Hardware-Aware Processing Pipeline",
        ]
        for i, ln in enumerate(lines):
            screen.blit(f_small.render(ln, True, DARK_GREEN), (70, 150 + i * 26))

        # Instructions
        msg = "Drag & drop a satellite image  —  or type a file path below"
        screen.blit(f_med.render(msg, True, WHITE), (scr_w // 2 - f_med.size(msg)[0] // 2, 310))

        # Input box
        bx = 80
        bw = scr_w - 160
        ir = pygame.Rect(bx, 355, bw, 34)
        pygame.draw.rect(screen, (15, 15, 40), ir)
        ic = CYAN if input_active else GRAY
        pygame.draw.rect(screen, ic, ir, 2, border_radius=4)
        if input_text:
            it = f_small.render(input_text[-80:], True, WHITE)
        else:
            it = f_small.render("Enter image file path…", True, DARK_GRAY)
        screen.blit(it, (bx + 10, 361))

        # Cursor blink
        if input_active and int(t * 2) % 2:
            cx = bx + 10 + f_small.size(input_text[-80:])[0]
            pygame.draw.line(screen, CYAN, (cx, 359), (cx, 385), 2)

        # Status
        if status_msg:
            screen.blit(f_small.render(status_msg, True, status_col),
                        (scr_w // 2 - f_small.size(status_msg)[0] // 2, 405))

        # Timer
        rem = max(0, 30 - (time.time() - load_t0))
        tc = RED if rem < 10 else YELLOW if rem < 20 else GREEN
        screen.blit(f_small.render(f"Timeout: {rem:.0f}s", True, tc),
                    (scr_w // 2 - 50, 440))

        # Processing mode indicator
        pm = f_hud.render(f"Processing Mode: {pipeline.display_mode}", True, DARK_CYAN)
        screen.blit(pm, (scr_w // 2 - pm.get_width() // 2, 475))

        # Bottom hint
        hint = "ENTER: load path  |  Ctrl+V: paste  |  ESC: quit"
        screen.blit(f_hud.render(hint, True, GRAY), (scr_w // 2 - f_hud.size(hint)[0] // 2, scr_h - 30))

    # ======================== DRAW: NODE MARKING ========================

    def draw_node_marking():
        # Image
        screen.blit(sat_surface, (0, 0))

        # Grid overlay
        if show_grid:
            gs = pygame.Surface((img_w, img_h), pygame.SRCALPHA)
            for gx in range(0, img_w, GRID_SPACING):
                pygame.draw.line(gs, (255, 255, 255, 30), (gx, 0), (gx, img_h))
            for gy in range(0, img_h, GRID_SPACING):
                pygame.draw.line(gs, (255, 255, 255, 30), (0, gy), (img_w, gy))
            screen.blit(gs, (0, 0))

        # Preview edges (k-nearest)
        if len(nodes) >= 2:
            g = build_knn_graph(nodes, K_NEIGHBORS)
            drawn = set()
            for nid, neighbours in g.items():
                for nb, _ in neighbours:
                    key = (min(nid, nb), max(nid, nb))
                    if key not in drawn:
                        drawn.add(key)
                        p1 = (int(nodes[nid][0]), int(nodes[nid][1]))
                        p2 = (int(nodes[nb][0]), int(nodes[nb][1]))
                        pygame.draw.line(screen, (80, 80, 80), p1, p2, 1)

        # Nodes
        for nid, pos in nodes.items():
            x, y = int(pos[0]), int(pos[1])
            nt = node_types.get(nid, "normal")
            if nt == "source":
                ar = int(18 + 6 * math.sin(anim_t * 3))
                as_ = pygame.Surface((ar * 2, ar * 2), pygame.SRCALPHA)
                pygame.draw.circle(as_, (0, 255, 80, 50), (ar, ar), ar)
                screen.blit(as_, (x - ar, y - ar))
                pygame.draw.circle(screen, GREEN, (x, y), 10)
                pygame.draw.circle(screen, WHITE, (x, y), 10, 2)
                screen.blit(f_small.render("S", True, WHITE), (x + 14, y - 8))
            elif nt == "dest":
                ar = int(18 + 6 * math.sin(anim_t * 3))
                as_ = pygame.Surface((ar * 2, ar * 2), pygame.SRCALPHA)
                pygame.draw.circle(as_, (255, 50, 50, 50), (ar, ar), ar)
                screen.blit(as_, (x - ar, y - ar))
                pygame.draw.circle(screen, RED, (x, y), 10)
                pygame.draw.circle(screen, WHITE, (x, y), 10, 2)
                screen.blit(f_small.render("D", True, WHITE), (x + 14, y - 8))
            else:
                pygame.draw.circle(screen, WHITE, (x, y), 7)
                pygame.draw.circle(screen, GRAY, (x, y), 7, 2)
                screen.blit(f_hud.render(str(nid), True, WHITE), (x + 10, y - 6))

        # Bottom controls bar
        bar = pygame.Surface((scr_w, HUD_HEIGHT), pygame.SRCALPHA)
        bar.fill((8, 8, 28, 225))
        screen.blit(bar, (0, scr_h - HUD_HEIGHT))
        pygame.draw.line(screen, CYAN, (0, scr_h - HUD_HEIGHT), (scr_w, scr_h - HUD_HEIGHT), 2)

        ctrls = [
            "LCLICK:Place", "D:Destination", "RCLICK:Undo",
            f"G:Grid({'ON' if show_grid else 'OFF'})",
            f"E:Snap({'ON' if edge_snap else 'OFF'})",
            "S:Save+Start",
            f"Nodes:{len(nodes)}",
        ]
        cx = 15
        for c in ctrls:
            t = f_hud.render(c, True, CYAN)
            screen.blit(t, (cx, scr_h - HUD_HEIGHT + 10))
            cx += t.get_width() + 18

        # Status
        if status_msg:
            sm = f_small.render(status_msg, True, status_col)
            screen.blit(sm, (scr_w // 2 - sm.get_width() // 2, scr_h - HUD_HEIGHT + 40))

        # Source/Dest info
        info_parts = []
        if source_id is not None:
            info_parts.append(f"SRC={source_id}")
        if dest_id is not None:
            info_parts.append(f"DST={dest_id}")
        if info_parts:
            info = "  |  ".join(info_parts)
            screen.blit(f_small.render(info, True, GREEN), (15, scr_h - HUD_HEIGHT + 55))

    # ======================== ENHANCED UI PANELS ========================

    def draw_pov_panel():
        """Drone POV camera feed panel with keypoint overlay (top-right).
        Uses percentage-based sizing and safe-zone opacity."""
        if drone is None or feat_ext is None:
            return None

        pw = clamp(int(scr_w * 0.25), 200, 360)
        ph = int(pw * 0.75)
        total_h = ph + s(55)

        # Use draggable panel position if available
        pov_panel = panel_mgr.get("pov")
        if pov_panel:
            pov_panel.width = pw
            pov_panel.height = total_h
            px = pov_panel.x
            py = pov_panel.y
        else:
            px = scr_w - pw - int(scr_w * 0.02)
            py = int(scr_h * 0.02)

        # Safe-zone opacity: reduce alpha when drone is near panel
        panel_alpha = PANEL_ALPHA
        if drone:
            dx, dy = drone.ipos
            safe_r = int(120 * ui_scale)
            # Check if drone is within safe radius of panel
            closest_x = clamp(dx, px, px + pw)
            closest_y = clamp(dy, py, py + total_h)
            dist_to_panel = euclidean_dist((dx, dy), (closest_x, closest_y))
            if dist_to_panel < safe_r:
                panel_alpha = 120

        # Panel background
        panel_s = pygame.Surface((pw, total_h), pygame.SRCALPHA)
        panel_s.fill((*PANEL_BG, panel_alpha))
        screen.blit(panel_s, (px, py))

        # Label
        lbl = f_hud.render("DRONE POV (LIVE)", True, NEON_GREEN)
        screen.blit(lbl, (px + 8, py + 3))
        # Live indicator dot
        dot_pulse = int(4 + 2 * math.sin(anim_t * 6))
        pygame.draw.circle(screen, RED, (px + pw - 12, py + 9), dot_pulse)

        # Separator
        pygame.draw.line(screen, (0, 80, 120), (px + 5, py + 18), (px + pw - 5, py + 18), 1)

        # POV image from localizer cache or live extraction
        pov_img = None
        if localizer and localizer.last_kp_img is not None:
            pov_img = localizer.last_kp_img
        elif feat_ext:
            try:
                vc, _ = feat_ext._crop(drone.true_pos)
                if vc.size > 0:
                    pov_img = cv2.cvtColor(vc, cv2.COLOR_GRAY2BGR)
            except Exception:
                pass

        if pov_img is not None:
            try:
                crop_h, crop_w = pov_img.shape[:2]
                inner_w = pw - 16
                inner_h = ph - 28
                sc = min(inner_w / max(crop_w, 1), inner_h / max(crop_h, 1))
                nw, nh = max(1, int(crop_w * sc)), max(1, int(crop_h * sc))
                resized = cv2.resize(pov_img, (nw, nh))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pov_surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                ix = px + (pw - nw) // 2
                iy = py + 22
                screen.blit(pov_surf, (ix, iy))

                # Scanline effect on POV
                scan_y = int(nh * ((anim_t * 0.5) % 1.0))
                scan_s = pygame.Surface((nw, 2), pygame.SRCALPHA)
                scan_s.fill((0, 255, 100, 35))
                screen.blit(scan_s, (ix, iy + scan_y))

                # Matching status overlay bar at bottom of POV image
                if pov_matcher_status:
                    bar_s = pygame.Surface((nw, 16), pygame.SRCALPHA)
                    bar_s.fill((0, 0, 0, 180))
                    screen.blit(bar_s, (ix, iy + nh - 16))
                    ms_txt = f_hud.render(pov_matcher_status, True, YELLOW)
                    screen.blit(ms_txt, (ix + 4, iy + nh - 14))

                # POV image border
                pygame.draw.rect(screen, (0, 80, 120), (ix - 1, iy - 1, nw + 2, nh + 2), 1)
            except Exception:
                pass

        # Confidence bar below POV
        bar_y = py + ph + 2
        loc_conf = smooth_confidence
        bar_col = confidence_color(loc_conf)
        draw_progress_bar(screen, px + 8, bar_y, pw - 16, 12, loc_conf, bar_col)
        conf_lbl = f_hud.render(f"CONFIDENCE {loc_conf*100:.0f}%", True, bar_col)
        screen.blit(conf_lbl, (px + 8, bar_y + 16))

        # Keypoint count
        kp_count = localizer.last_kp_count if localizer else 0
        kp_txt = f"KP:{kp_count}"
        kp_lbl = f_hud.render(kp_txt, True, DARK_CYAN)
        screen.blit(kp_lbl, (px + pw - f_hud.size(kp_txt)[0] - 8, bar_y + 16))

        # Panel border + brackets
        pygame.draw.rect(screen, CYAN, (px, py, pw, total_h), 1)
        draw_corner_brackets(screen, (px, py, pw, total_h), CYAN, 10, 1)

        return (px, py, pw, total_h)

    def draw_debug_panel(pov_bounds):
        """Localization debug panel (below POV on right side)."""
        if localizer is None:
            return
        px, py_pov, pw, pov_h = pov_bounds
        dpy = py_pov + pov_h + 8
        dph = 148

        # Background
        dp = pygame.Surface((pw, dph), pygame.SRCALPHA)
        dp.fill((*PANEL_BG, PANEL_ALPHA))
        screen.blit(dp, (px, dpy))

        # Title
        screen.blit(f_hud.render("LOCALIZATION DEBUG", True, CYAN), (px + 8, dpy + 4))
        pygame.draw.line(screen, (0, 80, 120), (px + 5, dpy + 18), (px + pw - 5, dpy + 18), 1)

        # Data rows
        ly = dpy + 24
        row_h = 16

        loc_conf = localizer.confidence
        lm = localizer.loc_mode
        lm_col = GREEN if lm == LOC_NORMAL else YELLOW if lm == LOC_SEARCHING else RED

        rows = [
            (f"EST POS    ({int(localizer.estimated_pos[0])},{int(localizer.estimated_pos[1])})", CYAN),
            (f"CONFIDENCE {loc_conf*100:.1f}%", confidence_color(loc_conf)),
            (f"LOC MODE   {lm}", lm_col),
            (f"MATCHER    {localizer.matcher_used}", WHITE),
            (f"RANSAC     {localizer.ransac_inliers_pct*100:.0f}%", WHITE),
            (f"SEARCH WIN {localizer.search_window_size}px", WHITE),
            (f"FRAME      {localizer.frame_counter}", DARK_GRAY),
        ]

        for txt, col in rows:
            screen.blit(f_hud.render(txt, True, col), (px + 10, ly))
            ly += row_h

        # Border
        pygame.draw.rect(screen, CYAN, (px, dpy, pw, dph), 1)
        draw_corner_brackets(screen, (px, dpy, pw, dph), DARK_CYAN, 8, 1)

    def draw_status_banner_display():
        """Dynamic status banner at top-center."""
        if status_banner_timer <= 0 or not status_banner_text:
            return

        # Fade alpha based on remaining time
        alpha = int(clamp(status_banner_timer / 0.5, 0, 1) * 220)

        txt = f_large.render(status_banner_text, True, status_banner_color)
        tw = txt.get_width()
        th = txt.get_height()
        bw = tw + 50
        bh = th + 16
        bx = scr_w // 2 - bw // 2
        by = 8

        bg = pygame.Surface((bw, bh), pygame.SRCALPHA)
        bg.fill((10, 12, 28, min(alpha, 200)))
        screen.blit(bg, (bx, by))

        pygame.draw.rect(screen, status_banner_color, (bx, by, bw, bh), 1, border_radius=4)
        draw_corner_brackets(screen, (bx, by, bw, bh), status_banner_color, 8, 1)

        screen.blit(txt, (bx + 25, by + 8))

    def draw_process_log():
        """Process flow log showing last 5 actions (bottom-left, above HUD)."""
        if not process_log:
            return

        log_w = 310
        num = len(process_log)
        log_h = num * 16 + 22

        # Use draggable panel position
        pl_panel = panel_mgr.get("process_log")
        if pl_panel:
            pl_panel.width = log_w
            pl_panel.height = log_h
            log_x = pl_panel.x
            log_y = pl_panel.y
        else:
            log_x = 12
            log_y = scr_h - HUD_HEIGHT - log_h - 8

        # Background
        bg = pygame.Surface((log_w, log_h), pygame.SRCALPHA)
        bg.fill((*PANEL_BG, 180))
        screen.blit(bg, (log_x, log_y))

        # Title
        screen.blit(f_hud.render("PROCESS LOG", True, DARK_CYAN), (log_x + 6, log_y + 2))

        # Entries (newest first)
        now = time.time()
        for i, (ts, msg, col) in enumerate(reversed(process_log)):
            age = now - ts
            fade = max(0.3, 1.0 - age / 12.0)
            faded_col = (int(col[0] * fade), int(col[1] * fade), int(col[2] * fade))
            ey = log_y + 18 + i * 16
            prefix = ">" if i == 0 else " "
            screen.blit(f_hud.render(f"{prefix} {msg}", True, faded_col), (log_x + 6, ey))

        # Border
        pygame.draw.rect(screen, (0, 80, 120), (log_x, log_y, log_w, log_h), 1)

    def draw_search_visualization():
        """Pulsing search window when localization confidence is low."""
        if drone is None or localizer is None:
            return
        if localizer.loc_mode == LOC_NORMAL:
            return

        dx, dy = drone.ipos
        sr = localizer.search_window_size
        p_val = pulse(anim_t, speed=2.0, lo=0.15, hi=0.45)
        alpha = int(p_val * 255)

        search_col = YELLOW if localizer.loc_mode == LOC_SEARCHING else RED

        # Pulsing translucent square
        search_surf = pygame.Surface((sr * 2, sr * 2), pygame.SRCALPHA)
        pygame.draw.rect(search_surf, (*search_col, int(alpha * 0.3)),
                         (0, 0, sr * 2, sr * 2))
        pygame.draw.rect(search_surf, (*search_col, alpha),
                         (0, 0, sr * 2, sr * 2), 2)
        screen.blit(search_surf, (dx - sr, dy - sr))

        # Label
        mode_lbl = "SEARCH MODE ACTIVE" if localizer.loc_mode == LOC_SEARCHING else "POSITION LOST"
        lbl_surf = f_hud.render(mode_lbl, True, search_col)
        # Background for label
        lbl_bg = pygame.Surface((lbl_surf.get_width() + 10, lbl_surf.get_height() + 4), pygame.SRCALPHA)
        lbl_bg.fill((0, 0, 0, 160))
        lbl_x = dx - lbl_surf.get_width() // 2
        lbl_y = dy - sr - 20
        screen.blit(lbl_bg, (lbl_x - 5, lbl_y - 2))
        screen.blit(lbl_surf, (lbl_x, lbl_y))

    def draw_target_arrow():
        """Arrow from drone pointing toward next target node."""
        if drone is None or not path or path_idx >= len(path):
            return
        tid = path[path_idx]
        if tid not in nodes:
            return

        dx, dy = drone.ipos
        tx, ty = int(nodes[tid][0]), int(nodes[tid][1])
        ddx = tx - dx
        ddy = ty - dy
        dist = math.sqrt(ddx * ddx + ddy * ddy)
        if dist < 30:
            return

        # Normalize and scale arrow
        arrow_len = min(55, dist * 0.25)
        ux, uy = ddx / dist, ddy / dist

        # Start slightly away from drone center
        sx = int(dx + ux * 18)
        sy = int(dy + uy * 18)
        ex = int(dx + ux * (18 + arrow_len))
        ey = int(dy + uy * (18 + arrow_len))

        # Color based on confidence
        loc_conf = localizer.confidence if localizer else confidence
        arr_col = confidence_color(loc_conf)

        # Shaft
        pygame.draw.line(screen, arr_col, (sx, sy), (ex, ey), 2)

        # Arrowhead
        perp_x, perp_y = -uy, ux
        head_sz = 7
        h1 = (int(ex - ux * head_sz + perp_x * head_sz),
              int(ey - uy * head_sz + perp_y * head_sz))
        h2 = (int(ex - ux * head_sz - perp_x * head_sz),
              int(ey - uy * head_sz - perp_y * head_sz))
        pygame.draw.polygon(screen, arr_col, [(ex, ey), h1, h2])

    def draw_performance_display():
        """EXPANDED performance monitoring panel — FPS, frame time,
        match time, localize time, processing mode, and FPS sparkline."""
        fps = clock.get_fps()

        pw_perf = 183
        ph_perf = 106

        # Use draggable panel position
        perf_p = panel_mgr.get("performance")
        if perf_p:
            perf_p.width = pw_perf
            perf_p.height = ph_perf
            perf_x = perf_p.x
            perf_y = perf_p.y
        else:
            perf_x = scr_w - 195
            perf_y = scr_h - HUD_HEIGHT - 110

        bg = pygame.Surface((pw_perf, ph_perf), pygame.SRCALPHA)
        bg.fill((*PANEL_BG, 200))
        screen.blit(bg, (perf_x, perf_y))

        # Title
        screen.blit(f_hud.render("PERFORMANCE MONITOR", True, CYAN), (perf_x + 6, perf_y + 2))
        pygame.draw.line(screen, (0, 80, 120), (perf_x + 5, perf_y + 16), (perf_x + 178, perf_y + 16), 1)

        # Row 1: FPS
        fps_col = GREEN if fps >= 50 else YELLOW if fps >= 30 else RED
        screen.blit(f_hud.render(f"FPS: {fps:.0f}", True, fps_col), (perf_x + 6, perf_y + 20))

        # Row 2: Frame time
        screen.blit(f_hud.render(f"FRAME: {frame_time_ms:.1f}ms", True, WHITE), (perf_x + 6, perf_y + 34))

        # Row 3: Match time
        screen.blit(f_hud.render(f"MATCH: {match_time_ms:.1f}ms", True, WHITE), (perf_x + 6, perf_y + 48))

        # Row 4: Localization time
        screen.blit(f_hud.render(f"LOCALIZE: {localize_time_ms:.1f}ms", True, WHITE), (perf_x + 6, perf_y + 62))

        # Row 5: Processing mode
        pm = pipeline.display_mode
        pm_col = GREEN if pm == PROC_MODE_GPU else CYAN if pm == PROC_MODE_NPU else YELLOW
        screen.blit(f_hud.render(f"MODE: {pm}", True, pm_col), (perf_x + 6, perf_y + 76))

        # FPS Sparkline (last 60 values)
        if len(fps_history) > 1:
            spark_x = perf_x + 100
            spark_y = perf_y + 20
            spark_w = 75
            spark_h = 14
            # Background for sparkline
            pygame.draw.rect(screen, (20, 20, 40), (spark_x, spark_y, spark_w, spark_h))
            max_fps = max(max(fps_history), 1)
            n = len(fps_history)
            step = spark_w / max(n - 1, 1)
            pts = []
            for i, fv in enumerate(fps_history):
                sx = int(spark_x + i * step)
                sy = int(spark_y + spark_h - (fv / max_fps) * spark_h)
                pts.append((sx, clamp(sy, spark_y, spark_y + spark_h)))
            if len(pts) >= 2:
                pygame.draw.lines(screen, fps_col, False, pts, 1)
            pygame.draw.rect(screen, DARK_CYAN, (spark_x, spark_y, spark_w, spark_h), 1)

        # Border
        pygame.draw.rect(screen, DARK_CYAN, (perf_x, perf_y, 183, 106), 1)
        draw_corner_brackets(screen, (perf_x, perf_y, 183, 106), DARK_CYAN, 8, 1)

    # ======================== DRAW: COMPASS + HEADING ========================

    def draw_compass():
        """Compass bar with heading indicator (top-center, below banner)."""
        if drone is None:
            return

        comp_w = 220
        comp_h = 28
        comp_x = scr_w // 2 - comp_w // 2
        comp_y = 42

        # Background
        bg = pygame.Surface((comp_w, comp_h), pygame.SRCALPHA)
        bg.fill((10, 12, 28, 170))
        screen.blit(bg, (comp_x, comp_y))

        # Heading angle in degrees (0=East, convert to compass: 0=North)
        hdeg = (-math.degrees(smooth_heading) + 90) % 360

        # Direction labels at fixed positions
        dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        angles = [0, 45, 90, 135, 180, 225, 270, 315]

        center_px = comp_w // 2
        px_per_deg = comp_w / 180.0  # show ±90° range

        for lbl, da in zip(dirs, angles):
            offset_deg = ((da - hdeg + 180) % 360) - 180  # diff from heading
            if -90 <= offset_deg <= 90:
                lx = int(comp_x + center_px + offset_deg * px_per_deg)
                is_main = lbl in ("N", "S", "E", "W")
                col = CYAN if is_main else DARK_CYAN
                txt = f_hud.render(lbl, True, col)
                screen.blit(txt, (lx - txt.get_width() // 2, comp_y + 3))
                if is_main:
                    pygame.draw.line(screen, col, (lx, comp_y + 18), (lx, comp_y + comp_h - 2), 1)

        # Center tick (current heading)
        cx = comp_x + center_px
        pygame.draw.polygon(screen, CYAN, [(cx, comp_y + comp_h), (cx - 4, comp_y + comp_h + 5), (cx + 4, comp_y + comp_h + 5)])

        # Heading degree readout
        hdeg_txt = f_hud.render(f"{hdeg:.0f}°", True, WHITE)
        screen.blit(hdeg_txt, (cx - hdeg_txt.get_width() // 2, comp_y + comp_h + 5))

        # Border
        pygame.draw.rect(screen, DARK_CYAN, (comp_x, comp_y, comp_w, comp_h), 1)

    # ======================== DRAW: CONFIDENCE HISTORY GRAPH ========================

    def draw_confidence_graph():
        """Small confidence history sparkline panel (left side, mid-height)."""
        if not confidence_history:
            return

        gw = 160
        gh = 48

        # Use draggable panel position
        conf_p = panel_mgr.get("confidence")
        if conf_p:
            conf_p.width = gw
            conf_p.height = gh
            gx = conf_p.x
            gy = conf_p.y
        else:
            gx = 12
            gy = scr_h // 2 - gh // 2

        # Background
        bg = pygame.Surface((gw, gh), pygame.SRCALPHA)
        bg.fill((10, 12, 28, 170))
        screen.blit(bg, (gx, gy))

        # Title
        screen.blit(f_hud.render("CONF HISTORY", True, DARK_CYAN), (gx + 4, gy + 2))

        # Graph area
        graph_y = gy + 14
        graph_h = gh - 18
        n = len(confidence_history)
        if n >= 2:
            step_x = (gw - 8) / max(n - 1, 1)
            pts = []
            for i, cv in enumerate(confidence_history):
                px = int(gx + 4 + i * step_x)
                py_pt = int(graph_y + graph_h - cv * graph_h)
                py_pt = clamp(py_pt, graph_y, graph_y + graph_h)
                pts.append((px, py_pt))

            # Draw segments colored by confidence
            for i in range(1, len(pts)):
                cv = confidence_history[i]
                col = GREEN if cv > 0.6 else YELLOW if cv > 0.3 else RED
                pygame.draw.line(screen, col, pts[i - 1], pts[i], 1)

        # Current value
        if confidence_history:
            cur = confidence_history[-1]
            cc = confidence_color(cur)
            cv_txt = f_hud.render(f"{cur*100:.0f}%", True, cc)
            screen.blit(cv_txt, (gx + gw - cv_txt.get_width() - 4, gy + 2))

        # Border
        pygame.draw.rect(screen, (0, 80, 120), (gx, gy, gw, gh), 1)

    # ======================== DRAW: TARGET LOCK ANIMATION ========================

    def draw_target_lock():
        """Shrinking circle animation on node confirmation/failure."""
        nonlocal target_lock_active, target_lock_timer

        if not target_lock_active:
            return

        if target_lock_timer <= 0:
            target_lock_active = False
            return

        # Progress: 1.0 → 0.0
        progress = target_lock_timer / target_lock_duration
        lx, ly = int(target_lock_pos[0]), int(target_lock_pos[1])

        # Shrinking ring
        max_r = 40
        min_r = 8
        r = int(min_r + (max_r - min_r) * progress)
        alpha = int(clamp(progress * 200, 0, 200))

        lock_surf = pygame.Surface((max_r * 2 + 4, max_r * 2 + 4), pygame.SRCALPHA)
        center = max_r + 2
        # Outer ring
        pygame.draw.circle(lock_surf, (*target_lock_color, alpha), (center, center), r, 3)
        # Inner crosshair
        if progress < 0.5:
            ch_len = int(6 + 4 * (1.0 - progress))
            pygame.draw.line(lock_surf, (*target_lock_color, alpha),
                             (center - ch_len, center), (center + ch_len, center), 1)
            pygame.draw.line(lock_surf, (*target_lock_color, alpha),
                             (center, center - ch_len), (center, center + ch_len), 1)

        screen.blit(lock_surf, (lx - center, ly - center))

        # "LOCKED" text on completion
        if progress < 0.3:
            ltxt = f_hud.render("LOCKED", True, target_lock_color)
            screen.blit(ltxt, (lx - ltxt.get_width() // 2, ly + max_r + 4))

    # ======================== DRAW: NAVIGATION ========================

    def draw_navigation():
        # Reveal surface
        compute_reveal()
        if reveal_surface_cache is not None:
            screen.blit(reveal_surface_cache, (0, 0))
        else:
            screen.blit(sat_surface, (0, 0))

        # Uncertainty Heatmap overlay
        if heatmap:
            heatmap.render(screen, pygame)

        # Path edges
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                n1, n2 = path[i], path[i + 1]
                if n1 not in nodes or n2 not in nodes:
                    continue
                p1 = (int(nodes[n1][0]), int(nodes[n1][1]))
                p2 = (int(nodes[n2][0]), int(nodes[n2][1]))
                if i < path_idx - 1:
                    pygame.draw.line(screen, DARK_GREEN, p1, p2, 2)
                elif i == path_idx - 1:
                    col = CYAN if int(anim_t * 6) % 2 else DARK_CYAN
                    pygame.draw.line(screen, col, p1, p2, 3)
                else:
                    pygame.draw.line(screen, DARK_GRAY, p1, p2, 1)

        # Nodes
        for nid, pos in nodes.items():
            x, y = int(pos[0]), int(pos[1])

            if nid in blocked:
                pygame.draw.circle(screen, DARK_RED, (x, y), 8)
                pygame.draw.line(screen, RED, (x - 5, y - 5), (x + 5, y + 5), 2)
                pygame.draw.line(screen, RED, (x + 5, y - 5), (x - 5, y + 5), 2)
                continue

            nt = node_types.get(nid, "normal")
            if nid in confirmed:
                pygame.draw.circle(screen, GREEN, (x, y), 7)
                pygame.draw.circle(screen, WHITE, (x, y), 7, 2)
            elif nid in skipped:
                pygame.draw.circle(screen, ORANGE, (x, y), 7)
            elif nt == "source":
                pygame.draw.circle(screen, GREEN, (x, y), 9)
                pygame.draw.circle(screen, WHITE, (x, y), 9, 2)
            elif nt == "dest":
                ar = int(14 + 5 * math.sin(anim_t * 4))
                as_ = pygame.Surface((ar * 2, ar * 2), pygame.SRCALPHA)
                pygame.draw.circle(as_, (255, 50, 50, 70), (ar, ar), ar)
                screen.blit(as_, (x - ar, y - ar))
                pygame.draw.circle(screen, RED, (x, y), 9)
            else:
                pygame.draw.circle(screen, (150, 150, 150), (x, y), 5)

            screen.blit(f_hud.render(str(nid), True, WHITE), (x + 10, y - 5))

        # Target node indicator
        if path and path_idx < len(path):
            tid = path[path_idx]
            if tid in nodes:
                tx, ty = int(nodes[tid][0]), int(nodes[tid][1])
                r = int(20 + 8 * math.sin(anim_t * 5))
                ts = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(ts, (0, 204, 255, 60), (r, r), r)
                screen.blit(ts, (tx - r, ty - r))
                pygame.draw.circle(screen, CYAN, (tx, ty), 12, 2)

        # Drone trail (alpha-fading, confidence-colored)
        if drone and len(drone.trail) > 1:
            tl = len(drone.trail)
            step = max(1, tl // 400)  # limit drawn segments
            for i in range(step, tl, step):
                ci = i if i < len(drone.conf_trail) else -1
                c = drone.conf_trail[ci] if ci >= 0 and ci < len(drone.conf_trail) else 0.5
                col = confidence_color(c)
                # Fade older segments
                age_frac = i / max(tl, 1)
                fade = max(80, int(255 * age_frac))
                faded_col = (min(col[0], fade), min(col[1], fade), min(col[2], fade))
                pygame.draw.line(screen, faded_col, drone.trail[i - step], drone.trail[i], 2)

        # Drone marker — ESTIMATED position (what the drone "thinks")
        if drone:
            dx, dy = drone.ipos  # estimated position
            # Confidence aura
            loc_conf = localizer.confidence if localizer else confidence
            cc = confidence_color(loc_conf)
            ar = int(22 + 8 * math.sin(anim_t * 5))
            aurasf = pygame.Surface((ar * 2, ar * 2), pygame.SRCALPHA)
            pygame.draw.circle(aurasf, (*cc, 45), (ar, ar), ar)
            screen.blit(aurasf, (dx - ar, dy - ar))

            # --- Soft shadow under drone ---
            shadow_surf = pygame.Surface((28, 28), pygame.SRCALPHA)
            pygame.draw.circle(shadow_surf, (0, 0, 0, 40), (14, 14), 12)
            screen.blit(shadow_surf, (dx - 14 + 3, dy - 14 + 4))

            # Crosshair at estimated position
            pygame.draw.line(screen, CYAN, (dx - 16, dy), (dx - 6, dy), 1)
            pygame.draw.line(screen, CYAN, (dx + 6, dy), (dx + 16, dy), 1)
            pygame.draw.line(screen, CYAN, (dx, dy - 16), (dx, dy - 6), 1)
            pygame.draw.line(screen, CYAN, (dx, dy + 6), (dx, dy + 16), 1)

            # Quadcopter shape with heading orientation
            a = smooth_heading
            sz = 12
            arm_len = sz * 0.8

            # 4 arms at 45° offsets from heading
            for arm_angle_offset in [0.785, 2.356, 3.927, 5.498]:  # π/4 intervals
                aa = a + arm_angle_offset
                ax_end = dx + int(arm_len * math.cos(aa))
                ay_end = dy + int(arm_len * math.sin(aa))
                pygame.draw.line(screen, CYAN, (dx, dy), (ax_end, ay_end), 2)
                # Rotor circle at arm tip
                pygame.draw.circle(screen, WHITE, (ax_end, ay_end), 3, 1)

            # Center body
            pygame.draw.circle(screen, CYAN, (dx, dy), 4)
            pygame.draw.circle(screen, WHITE, (dx, dy), 4, 1)

            # Heading arrow (front direction)
            arrow_len = sz + 4
            ax_tip = dx + int(arrow_len * math.cos(a))
            ay_tip = dy + int(arrow_len * math.sin(a))
            pygame.draw.line(screen, GREEN, (dx, dy), (ax_tip, ay_tip), 2)
            # Arrowhead
            perp_x, perp_y = -math.sin(a), math.cos(a)
            h1 = (int(ax_tip - math.cos(a) * 4 + perp_x * 3),
                  int(ay_tip - math.sin(a) * 4 + perp_y * 3))
            h2 = (int(ax_tip - math.cos(a) * 4 - perp_x * 3),
                  int(ay_tip - math.sin(a) * 4 - perp_y * 3))
            pygame.draw.polygon(screen, GREEN, [(ax_tip, ay_tip), h1, h2])

            # --- TRUE position ghost overlay (always shown for demo clarity) ---
            tx, ty = drone.true_ipos
            # True position ghost dot (faint red)
            ghost_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(ghost_surf, (255, 60, 60, 60), (10, 10), 7)
            screen.blit(ghost_surf, (tx - 10, ty - 10))
            pygame.draw.circle(screen, (255, 60, 60), (tx, ty), 3)

            # Error line (dashed) connecting true → estimated
            error_dist = euclidean_dist(drone.true_pos, drone.pos)
            if error_dist > 3:
                line_surf = pygame.Surface((img_w, img_h), pygame.SRCALPHA)
                dash_len = 8
                gap_len = 5
                ldx = dx - tx
                ldy = dy - ty
                total = math.sqrt(ldx * ldx + ldy * ldy)
                if total > 0:
                    ux, uy = ldx / total, ldy / total
                    drawn_d = 0.0
                    while drawn_d < total:
                        seg_end = min(drawn_d + dash_len, total)
                        sx_s = int(tx + ux * drawn_d)
                        sy_s = int(ty + uy * drawn_d)
                        sx_e = int(tx + ux * seg_end)
                        sy_e = int(ty + uy * seg_end)
                        pygame.draw.line(line_surf, (255, 200, 0, 100),
                                         (sx_s, sy_s), (sx_e, sy_e), 1)
                        drawn_d += dash_len + gap_len
                screen.blit(line_surf, (0, 0))

                # Error label
                err_txt = f_hud.render(f"{error_dist:.0f}px", True, YELLOW)
                mid_x = (dx + tx) // 2
                mid_y = (dy + ty) // 2
                screen.blit(err_txt, (mid_x + 5, mid_y - 8))

            # --- Extended debug (toggled with T key) ---
            if show_true_pos and localizer:
                search_r = SEARCH_WINDOW_SEARCH if localizer.loc_mode != LOC_NORMAL else SEARCH_WINDOW_NORMAL
                sw_surf = pygame.Surface((search_r * 2 + 4, search_r * 2 + 4), pygame.SRCALPHA)
                n_segs = 36
                for seg_i in range(n_segs):
                    if seg_i % 2 == 0:
                        a1 = (seg_i / n_segs) * 2 * math.pi
                        a2 = ((seg_i + 1) / n_segs) * 2 * math.pi
                        cx1 = int(search_r + 2 + search_r * math.cos(a1))
                        cy1 = int(search_r + 2 + search_r * math.sin(a1))
                        cx2 = int(search_r + 2 + search_r * math.cos(a2))
                        cy2 = int(search_r + 2 + search_r * math.sin(a2))
                        pygame.draw.line(sw_surf, (0, 204, 255, 50),
                                         (cx1, cy1), (cx2, cy2), 1)
                screen.blit(sw_surf, (dx - search_r - 2, dy - search_r - 2))

        # Search mode pulsing visualization
        draw_search_visualization()

        # Target direction arrow
        draw_target_arrow()

        # Flash effects
        if flash_t > 0 and flash_col:
            fs = pygame.Surface((img_w, img_h), pygame.SRCALPHA)
            fa = int(clamp(flash_t * 80, 0, 100))
            fs.fill((*flash_col, fa))
            screen.blit(fs, (0, 0))

        # Replanning overlay
        if replan_flash > 0:
            rt = f_large.render("ADAPTIVE REPLANNING ACTIVE", True, YELLOW)
            rx = scr_w // 2 - rt.get_width() // 2
            ry = img_h // 2 - 20
            bx = pygame.Surface((rt.get_width() + 40, rt.get_height() + 20), pygame.SRCALPHA)
            bx.fill((0, 0, 0, 160))
            screen.blit(bx, (rx - 20, ry - 10))
            screen.blit(rt, (rx, ry))

        # Failure Prediction Warning
        draw_failure_prediction_warning()

        # Minimap (repositioned lower-right)
        draw_minimap()

        # Side panels
        pov_bounds = draw_pov_panel()
        if pov_bounds:
            draw_debug_panel(pov_bounds)

        # HUD bar
        draw_hud()

        # Overlay panels
        draw_status_banner_display()
        draw_compass()
        draw_process_log()
        draw_decision_panel()
        draw_intelligence_overlay()
        draw_confidence_graph()
        draw_target_lock()
        draw_performance_display()

    def draw_failure_prediction_warning():
        """Draw failure prediction warning overlay when approaching low-feature regions."""
        if failure_predictor is None or not failure_predictor.warning_active:
            return
        if failure_predictor.warning_rect is None:
            return

        rx, ry, rw, rh = failure_predictor.warning_rect

        # Pulsing red outline around warned region
        p_val = pulse(anim_t, speed=4.0, lo=0.3, hi=1.0)
        alpha = int(p_val * 180)

        warn_surf = pygame.Surface((rw + 8, rh + 8), pygame.SRCALPHA)
        pygame.draw.rect(warn_surf, (255, 60, 60, int(alpha * 0.3)),
                         (0, 0, rw + 8, rh + 8))
        pygame.draw.rect(warn_surf, (255, 60, 60, alpha),
                         (0, 0, rw + 8, rh + 8), 3)
        screen.blit(warn_surf, (rx - 4, ry - 4))

        # Warning text above the region
        warn_txt = f_small.render("\u26a0\ufe0f LOW VISUAL CONFIDENCE REGION", True, RED)
        txt_bg = pygame.Surface((warn_txt.get_width() + 14, warn_txt.get_height() + 6), pygame.SRCALPHA)
        txt_bg.fill((0, 0, 0, 200))
        txt_x = rx + rw // 2 - warn_txt.get_width() // 2
        txt_y = ry - 28
        screen.blit(txt_bg, (txt_x - 7, txt_y - 3))
        screen.blit(warn_txt, (txt_x, txt_y))

    def draw_decision_panel():
        """Decision Explanation System — rolling log panel (left side, above process log)."""
        if not decision_explainer.entries:
            return

        dp_w = 310
        num = len(decision_explainer.entries)
        dp_h = num * 16 + 26

        # Use draggable panel position
        dec_p = panel_mgr.get("decision")
        if dec_p:
            dec_p.width = dp_w
            dec_p.height = dp_h
            dp_x = dec_p.x
            dp_y = dec_p.y
        else:
            dp_x = 12
            dp_y = scr_h - HUD_HEIGHT - dp_h - 120

        # Avoid negative y
        if dp_y < 10:
            dp_y = 10

        # Background
        bg = pygame.Surface((dp_w, dp_h), pygame.SRCALPHA)
        bg.fill((*PANEL_BG, 190))
        screen.blit(bg, (dp_x, dp_y))

        # Title
        title_txt = f_hud.render("DECISION EXPLANATION", True, AMBER)
        screen.blit(title_txt, (dp_x + 6, dp_y + 3))
        pygame.draw.line(screen, (200, 140, 0, 80), (dp_x + 5, dp_y + 17),
                         (dp_x + dp_w - 5, dp_y + 17), 1)

        # Entries (newest first)
        now = time.time()
        for i, (ts, msg, col) in enumerate(reversed(decision_explainer.entries)):
            age = now - ts
            fade = max(0.4, 1.0 - age / 15.0)
            faded_col = (int(col[0] * fade), int(col[1] * fade), int(col[2] * fade))
            ey = dp_y + 20 + i * 16
            prefix = "\u25b6" if i == 0 else " "
            screen.blit(f_hud.render(f"{prefix} {msg}", True, faded_col), (dp_x + 6, ey))

        # Border
        pygame.draw.rect(screen, AMBER, (dp_x, dp_y, dp_w, dp_h), 1)
        draw_corner_brackets(screen, (dp_x, dp_y, dp_w, dp_h), AMBER, 8, 1)

    def draw_intelligence_overlay():
        """Intelligence status overlay — vision mode, learning mode, failure prediction."""
        ov_x = 12
        ov_y = 10
        ov_w = 230
        ov_h = 56

        # Background
        bg = pygame.Surface((ov_w, ov_h), pygame.SRCALPHA)
        bg.fill((*PANEL_BG, 180))
        screen.blit(bg, (ov_x, ov_y))

        # Vision Mode
        vm = vision_degradation.mode if vision_degradation else "CLEAR"
        vm_col = GREEN if vm == "CLEAR" else YELLOW if vm == "NOISY" else ORANGE if vm == "BLUR" else RED
        screen.blit(f_hud.render(f"Vision: {vm}", True, vm_col), (ov_x + 6, ov_y + 4))

        # Learning Mode
        lm_count = learning_memory.penalized_count if learning_memory else 0
        lm_txt = f"Learning: ACTIVE ({lm_count} regions)"
        lm_col = GREEN if lm_count == 0 else YELLOW if lm_count < 3 else ORANGE
        screen.blit(f_hud.render(lm_txt, True, lm_col), (ov_x + 6, ov_y + 20))

        # Failure Prediction
        fp_active = failure_predictor and failure_predictor.warning_active
        fp_txt = "Prediction: \u26a0 WARNING" if fp_active else "Prediction: OK"
        fp_col = RED if fp_active else GREEN
        screen.blit(f_hud.render(fp_txt, True, fp_col), (ov_x + 6, ov_y + 36))

        # Border
        pygame.draw.rect(screen, (0, 80, 120), (ov_x, ov_y, ov_w, ov_h), 1)

    def draw_minimap():
        if minimap_surf is None:
            return
        mm_x = scr_w - MINIMAP_W - 10
        mm_y = 10
        mm_h = minimap_h

        # Background
        bg = pygame.Surface((MINIMAP_W + 4, mm_h + 4), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        screen.blit(bg, (mm_x - 2, mm_y - 2))
        screen.blit(minimap_surf, (mm_x, mm_y))

        sx = MINIMAP_W / img_w
        sy = mm_h / img_h

        # Path
        if path and len(path) > 1:
            pts = [(int(mm_x + nodes[p][0] * sx), int(mm_y + nodes[p][1] * sy))
                   for p in path if p in nodes]
            if len(pts) >= 2:
                pygame.draw.lines(screen, CYAN, False, pts, 1)

        # Drone dot — ESTIMATED position (primary, green)
        if drone:
            mdx = int(mm_x + drone.est_x * sx)
            mdy = int(mm_y + drone.est_y * sy)
            pygame.draw.circle(screen, GREEN, (mdx, mdy), 3)

            # Debug: TRUE position (faint red dot)
            if show_true_pos:
                tdx = int(mm_x + drone.x * sx)
                tdy = int(mm_y + drone.y * sy)
                pygame.draw.circle(screen, (255, 60, 60), (tdx, tdy), 2)

        # Border
        pygame.draw.rect(screen, CYAN, (mm_x - 2, mm_y - 2, MINIMAP_W + 4, mm_h + 4), 1)
        # Minimap label with localization mode indicator
        loc_label = ""
        if localizer:
            loc_label = f"  [{localizer.loc_mode}]"
        screen.blit(f_hud.render(f"MINIMAP{loc_label}", True, CYAN), (mm_x, mm_y + mm_h + 4))

    def draw_hud():
        hy = scr_h - HUD_HEIGHT
        # Background
        bar = pygame.Surface((scr_w, HUD_HEIGHT), pygame.SRCALPHA)
        bar.fill((5, 5, 25, 235))
        screen.blit(bar, (0, hy))
        pygame.draw.line(screen, CYAN, (0, hy), (scr_w, hy), 2)

        y1 = hy + 8
        y2 = hy + 28
        y3 = hy + 48
        cw = scr_w // 7  # wider layout to fit localization info

        # Col 1: Estimated Position (what the drone believes)
        if drone:
            screen.blit(f_hud.render(f"EST POS ({int(drone.est_x)},{int(drone.est_y)})", True, CYAN), (10, y1))
            # Show true pos + error only in debug mode
            if show_true_pos:
                err = euclidean_dist(drone.true_pos, drone.pos)
                screen.blit(f_hud.render(f"TRUE ({int(drone.x)},{int(drone.y)}) Err:{err:.0f}px", True, (255, 100, 100)), (10, y2))
            else:
                # Status
                st_txt = sub_state or state
                sc = GREEN if "COMPLETE" in st_txt else CYAN if "MOVING" in st_txt else YELLOW if "RETRY" in st_txt or "MATCH" in st_txt else RED if "REPLAN" in st_txt else WHITE
                screen.blit(f_hud.render(f"STATUS  {st_txt}", True, sc), (10, y2))
        else:
            screen.blit(f_hud.render("EST POS (---)", True, CYAN), (10, y1))

        # Col 2: Node target
        if path and path_idx < len(path):
            screen.blit(f_hud.render(f"TARGET  Node {path[path_idx]}", True, WHITE), (cw, y1))
            screen.blit(f_hud.render(f"STEP    {path_idx}/{len(path)-1}", True, WHITE), (cw, y2))
        else:
            screen.blit(f_hud.render("TARGET  ---", True, WHITE), (cw, y1))

        # Col 3: Localization Confidence + bar
        loc_conf = localizer.confidence if localizer else confidence
        lcc = confidence_color(loc_conf)
        screen.blit(f_hud.render(f"LOC CONF  {loc_conf*100:.1f}%", True, lcc), (cw * 2, y1))
        draw_progress_bar(screen, cw * 2, y2 + 2, 120, 12, loc_conf, lcc)

        # Col 4: Localization Mode + Processing Mode
        if localizer:
            lm = localizer.loc_mode
            lm_col = GREEN if lm == LOC_NORMAL else YELLOW if lm == LOC_SEARCHING else RED
            screen.blit(f_hud.render(f"LOC MODE  {lm}", True, lm_col), (cw * 3, y1))
            speed_txt = f"SPEED  {localizer.get_speed():.1f}px/f"
            screen.blit(f_hud.render(speed_txt, True, WHITE), (cw * 3, y2))
        else:
            mu = matcher.matcher_used if matcher else "N/A"
            screen.blit(f_hud.render(f"MATCHER  {mu}", True, WHITE), (cw * 3, y1))
            mc = YELLOW if accurate_mode else GREEN
            screen.blit(f_hud.render(f"MODE  {'ACCURATE' if accurate_mode else 'FAST'}", True, mc), (cw * 3, y2))

        # Col 5: Match confidence bar + Timer
        cc = confidence_color(confidence)
        screen.blit(f_hud.render(f"MATCH  {confidence*100:.1f}%", True, cc), (cw * 4, y1))
        # Timer
        elapsed = (mission_t1 if mission_t1 else time.time()) - mission_t0 if mission_t0 else 0
        screen.blit(f_hud.render(f"TIME  {elapsed:.1f}s", True, WHITE), (cw * 4, y2))

        # Col 6: Progress + Processing Mode
        if path:
            prog = path_idx / max(len(path) - 1, 1)
            screen.blit(f_hud.render(f"PROGRESS  {prog*100:.0f}%", True, GREEN), (cw * 5, y1))
        screen.blit(f_hud.render(f"✓{len(confirmed)} ✗{len(skipped)}", True, WHITE), (cw * 5, y2))

        # Col 7: Processing Mode + Debug
        pm = pipeline.display_mode
        pm_col = GREEN if pm == PROC_MODE_GPU else CYAN if pm == PROC_MODE_NPU else YELLOW
        screen.blit(f_hud.render(f"PROC: {pm}", True, pm_col), (cw * 6, y1))
        dbg_col = GREEN if show_true_pos else DARK_GRAY
        screen.blit(f_hud.render(f"DEBUG {'ON' if show_true_pos else 'OFF'}", True, dbg_col), (cw * 6, y2))

        # Bottom row: shortcuts + replan count
        replan_info = ""
        if planner:
            replan_info = f" | REPLANS: {planner.replan_count}"
        vm = vision_degradation.mode if vision_degradation else "CLEAR"
        hints = f"M:Mode | T:Debug | V:Vision({vm}) | ESC:Quit{replan_info}"
        screen.blit(f_hud.render(hints, True, DARK_GRAY), (10, y3))

        # Corner brackets on HUD
        draw_corner_brackets(screen, (0, hy, scr_w, HUD_HEIGHT), DARK_CYAN, 12, 1)

    # ======================== DRAW: MISSION COMPLETE ========================

    def draw_mission_complete():
        screen.fill((5, 5, 18))
        if sat_surface:
            ds = sat_surface.copy()
            ds.set_alpha(40)
            screen.blit(ds, (0, 0))

        t = time.time()
        glow = int(200 + 55 * math.sin(t * 3))

        pw, ph = 520, 520
        px = scr_w // 2 - pw // 2
        py = scr_h // 2 - ph // 2

        panel = pygame.Surface((pw, ph), pygame.SRCALPHA)
        panel.fill((8, 8, 35, 235))
        screen.blit(panel, (px, py))
        pygame.draw.rect(screen, (0, glow, 255), (px, py, pw, ph), 2, border_radius=8)
        draw_corner_brackets(screen, (px, py, pw, ph), CYAN, 18, 2)

        # Title
        title = f_title.render("MISSION COMPLETE", True, GREEN)
        screen.blit(title, (scr_w // 2 - title.get_width() // 2, py + 20))

        # Divider
        pygame.draw.line(screen, DARK_CYAN, (px + 30, py + 55), (px + pw - 30, py + 55), 1)

        # --- Compute Mission Efficiency Score ---
        elapsed = mission_t1 - mission_t0 if mission_t1 else 0
        total_nodes = len(path) if path else 0
        replans = planner.replan_count if planner else 0
        failed_count = len(skipped)

        # Planned distance (sum of straight-line distances between path nodes)
        planned_distance = 0.0
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                if path[i] in nodes and path[i + 1] in nodes:
                    planned_distance += euclidean_dist(nodes[path[i]], nodes[path[i + 1]])

        # Actual distance (sum of trail segments)
        actual_distance = 0.0
        if drone and len(drone.trail) > 1:
            step = max(1, len(drone.trail) // 500)  # sample to avoid huge loop
            for i in range(step, len(drone.trail), step):
                actual_distance += euclidean_dist(drone.trail[i - step], drone.trail[i])

        # Efficiency formula
        if actual_distance > 0:
            efficiency = (planned_distance / actual_distance) * 100.0
        else:
            efficiency = 100.0

        # Penalize failures
        efficiency -= failed_count * 2.0

        # Clamp to [0, 100]
        efficiency = clamp(efficiency, 0.0, 100.0)

        # Efficiency color
        if efficiency >= 80:
            eff_col = GREEN
        elif efficiency >= 50:
            eff_col = YELLOW
        else:
            eff_col = RED

        # --- LARGE EFFICIENCY DISPLAY ---
        eff_txt = f_title.render(f"Mission Efficiency: {efficiency:.1f}%", True, eff_col)
        screen.blit(eff_txt, (scr_w // 2 - eff_txt.get_width() // 2, py + 65))

        # Efficiency bar
        bar_x = px + 40
        bar_w = pw - 80
        bar_y = py + 100
        draw_progress_bar(screen, bar_x, bar_y, bar_w, 16, efficiency / 100.0, eff_col)

        # Divider
        pygame.draw.line(screen, DARK_CYAN, (px + 30, py + 125), (px + pw - 30, py + 125), 1)

        # Stats
        stats = [
            (f"Time Taken:        {elapsed:.1f}s", WHITE),
            (f"Path Nodes:        {total_nodes}", WHITE),
            (f"Confirmed:         {len(confirmed)}", GREEN),
            (f"Failures:          {failed_count}", RED if failed_count > 0 else GREEN),
            (f"Blocked:           {len(blocked)}", RED if len(blocked) > 0 else WHITE),
            (f"Replans:           {replans}", YELLOW if replans > 0 else WHITE),
            (f"Planned Distance:  {planned_distance:.0f}px", CYAN),
            (f"Actual Distance:   {actual_distance:.0f}px", CYAN),
            (f"Processing:        {pipeline.display_mode}", CYAN),
            (f"Vision Mode:       {vision_degradation.mode}", WHITE),
            (f"Regions Learned:   {learning_memory.penalized_count}", YELLOW),
        ]
        for i, (txt, col) in enumerate(stats):
            screen.blit(f_small.render(txt, True, col), (px + 40, py + 140 + i * 26))

        # Options
        opt_r = f_small.render("[R]  Replay Mission          [ESC]  Exit", True, GRAY)
        screen.blit(opt_r, (scr_w // 2 - opt_r.get_width() // 2, py + ph - 30))

    # ======================== DRAW: ENHANCED REPLAY ========================

    def draw_replay():
        """Enhanced replay with smooth interpolation, event markers,
        timeline bar, speed control, and path quality visualization."""
        screen.blit(sat_surface, (0, 0))

        rs = replay_system

        # Draw full trail up to current position with PATH QUALITY coloring
        if rs.trail and rs.current_idx > 0:
            end = min(rs.current_idx + 1, len(rs.trail))
            for i in range(1, end):
                c = rs.get_conf_at(i)
                # Path quality: green (>0.6), yellow (0.3-0.6), red (<0.3)
                if c >= 0.6:
                    col = GREEN
                elif c >= 0.3:
                    col = YELLOW
                else:
                    col = RED
                pygame.draw.line(screen, col, rs.trail[i - 1], rs.trail[i], 3)

        # Draw EVENT MARKERS on trail
        for idx, evt in rs.events.items():
            if idx < len(rs.trail) and idx <= rs.current_idx:
                ex, ey = rs.trail[idx]
                if evt == REPLAY_EVENT_FAIL:
                    # Red flash marker
                    flash_s = pygame.Surface((24, 24), pygame.SRCALPHA)
                    pulse_r = int(12 + 4 * math.sin(anim_t * 8))
                    pygame.draw.circle(flash_s, (255, 60, 60, 120), (12, 12), pulse_r)
                    screen.blit(flash_s, (ex - 12, ey - 12))
                    pygame.draw.circle(screen, RED, (ex, ey), 5)
                    screen.blit(f_hud.render("FAIL", True, RED), (ex + 8, ey - 6))
                elif evt == REPLAY_EVENT_RETRY:
                    # Yellow marker
                    flash_s = pygame.Surface((20, 20), pygame.SRCALPHA)
                    pygame.draw.circle(flash_s, (255, 200, 0, 100), (10, 10), 10)
                    screen.blit(flash_s, (ex - 10, ey - 10))
                    pygame.draw.circle(screen, YELLOW, (ex, ey), 4)
                    screen.blit(f_hud.render("RETRY", True, YELLOW), (ex + 8, ey - 6))
                elif evt == REPLAY_EVENT_REPLAN:
                    # Blue marker
                    flash_s = pygame.Surface((24, 24), pygame.SRCALPHA)
                    pulse_r = int(12 + 4 * math.sin(anim_t * 6))
                    pygame.draw.circle(flash_s, (0, 120, 255, 120), (12, 12), pulse_r)
                    screen.blit(flash_s, (ex - 12, ey - 12))
                    pygame.draw.circle(screen, BLUE, (ex, ey), 5)
                    screen.blit(f_hud.render("REPLAN", True, BLUE), (ex + 8, ey - 6))

        # Animated drone dot (SMOOTHLY INTERPOLATED position)
        if rs.trail:
            rx, ry = rs.get_interpolated_pos()
            # Glow
            glow_s = pygame.Surface((30, 30), pygame.SRCALPHA)
            glow_r = int(15 + 4 * math.sin(anim_t * 5))
            pygame.draw.circle(glow_s, (0, 204, 255, 50), (15, 15), glow_r)
            screen.blit(glow_s, (rx - 15, ry - 15))
            pygame.draw.circle(screen, CYAN, (rx, ry), 8)
            pygame.draw.circle(screen, WHITE, (rx, ry), 8, 2)

        # Nodes
        for nid, pos in nodes.items():
            x, y = int(pos[0]), int(pos[1])
            nt = node_types.get(nid, "normal")
            if nt == "source":
                pygame.draw.circle(screen, GREEN, (x, y), 7)
            elif nt == "dest":
                pygame.draw.circle(screen, RED, (x, y), 7)
            else:
                pygame.draw.circle(screen, WHITE, (x, y), 4)

        # Header label with speed + pause indicator
        speed_txt = f"{rs.speed}×"
        pause_txt = "  ⏸ PAUSED" if rs.paused else ""
        rl = f_large.render(f"▶  REPLAY  ({speed_txt} Speed){pause_txt}", True, CYAN)
        bg = pygame.Surface((rl.get_width() + 30, rl.get_height() + 10), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        screen.blit(bg, (scr_w // 2 - rl.get_width() // 2 - 15, 15))
        screen.blit(rl, (scr_w // 2 - rl.get_width() // 2, 20))

        # Timestamp display
        cur_pts = int(rs.position)
        total_pts = max(len(rs.trail) - 1, 1)
        ts_txt = f"Frame: {cur_pts} / {total_pts}"
        ts_r = f_hud.render(ts_txt, True, WHITE)
        screen.blit(ts_r, (scr_w // 2 - ts_r.get_width() // 2, 45))

        # Path quality legend
        legend_y = 62
        legend_items = [("HIGH CONF", GREEN), ("MED CONF", YELLOW), ("LOW CONF", RED)]
        lx = scr_w // 2 - 120
        for ltxt, lcol in legend_items:
            pygame.draw.rect(screen, lcol, (lx, legend_y, 12, 8))
            screen.blit(f_hud.render(ltxt, True, lcol), (lx + 16, legend_y - 2))
            lx += 90

        # TIMELINE BAR at bottom
        tl_h = 24
        tl_y = scr_h - tl_h - 10
        tl_x = 40
        tl_w = scr_w - 80

        # Timeline background
        tl_bg = pygame.Surface((tl_w, tl_h), pygame.SRCALPHA)
        tl_bg.fill((10, 12, 28, 220))
        screen.blit(tl_bg, (tl_x, tl_y))

        # Timeline progress fill
        prog = rs.get_progress()
        fill_w = int(tl_w * prog)
        if fill_w > 0:
            pygame.draw.rect(screen, (0, 80, 160), (tl_x, tl_y, fill_w, tl_h))

        # Event markers on timeline
        if rs.trail:
            for idx, evt in rs.events.items():
                mx = tl_x + int((idx / max(len(rs.trail) - 1, 1)) * tl_w)
                if evt == REPLAY_EVENT_FAIL:
                    pygame.draw.line(screen, RED, (mx, tl_y), (mx, tl_y + tl_h), 2)
                elif evt == REPLAY_EVENT_RETRY:
                    pygame.draw.line(screen, YELLOW, (mx, tl_y), (mx, tl_y + tl_h), 2)
                elif evt == REPLAY_EVENT_REPLAN:
                    pygame.draw.line(screen, BLUE, (mx, tl_y), (mx, tl_y + tl_h), 2)

        # Playhead
        ph_x = tl_x + fill_w
        pygame.draw.rect(screen, WHITE, (ph_x - 1, tl_y - 3, 3, tl_h + 6))

        # Timeline border
        pygame.draw.rect(screen, CYAN, (tl_x, tl_y, tl_w, tl_h), 1, border_radius=2)

        # Timeline label
        pct_txt = f"{prog*100:.0f}%"
        screen.blit(f_hud.render(pct_txt, True, CYAN), (tl_x + tl_w + 6, tl_y + 4))

        # Speed control hint
        spd_hint = f_hud.render("SPACE: Pause | ←→: Seek | 0: 0.25× | 1: 0.5× | 2: 1× | 3: 2×", True, GRAY)
        screen.blit(spd_hint, (tl_x, tl_y - 16))

    # ======================== MAIN LOOP ========================

    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        frame_time_ms = dt * 1000.0
        anim_t += dt
        if flash_t > 0:
            flash_t -= dt
        if replan_flash > 0:
            replan_flash -= dt
        if status_banner_timer > 0:
            status_banner_timer -= dt

        # Smooth confidence interpolation
        target_conf = localizer.confidence if localizer else confidence
        smooth_confidence = lerp(smooth_confidence, target_conf, SMOOTH_FACTOR)

        # Smooth heading interpolation (prevents sharp compass jumps)
        if drone:
            target_heading = drone.heading
            # Handle angle wrapping for smooth interpolation
            diff = target_heading - smooth_heading
            while diff > math.pi:
                diff -= 2 * math.pi
            while diff < -math.pi:
                diff += 2 * math.pi
            smooth_heading += diff * 0.1  # smooth turn rate

        # Confidence history graph (sample every 3 frames)
        conf_hist_frame += 1
        if conf_hist_frame % 3 == 0:
            confidence_history.append(smooth_confidence)
            if len(confidence_history) > 50:
                confidence_history.pop(0)

        # Target lock animation timer
        if target_lock_active:
            target_lock_timer -= dt
            if target_lock_timer <= 0:
                target_lock_active = False

        # Camera follow (smooth tracking)
        if drone and cam_active:
            cam_x = lerp(cam_x, drone.est_x - scr_w // 2, 0.05)
            cam_y = lerp(cam_y, drone.est_y - scr_h // 2, 0.05)

        # Sync matcher status from PatternMatcher / VisualLocalizer
        if matcher:
            pov_matcher_status = matcher.match_stage
        elif localizer:
            pov_matcher_status = f"{localizer.matcher_used} ACTIVE"

        # FPS history for sparkline
        fps_history.append(clock.get_fps())
        if len(fps_history) > 60:
            fps_history.pop(0)

        # Update timing metrics
        if matcher:
            match_time_ms = matcher.last_match_time_ms
        if localizer:
            localize_time_ms = localizer.last_localize_time_ms

        # ---- Events ----
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            # Drag & drop
            elif ev.type == pygame.DROPFILE if hasattr(pygame, "DROPFILE") else -1:
                if state == STATE_LOADING:
                    fp = ev.file
                    ok, msg = load_image(fp)
                    set_status(msg, GREEN if ok else RED)
                    if ok:
                        state = STATE_MODE_SELECT

            # Keyboard
            elif ev.type == pygame.KEYDOWN:

                # ---- LOADING state keys ----
                if state == STATE_LOADING:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_RETURN:
                        fp = input_text.strip().strip('"').strip("'")
                        if fp and os.path.isfile(fp):
                            ok, msg = load_image(fp)
                            set_status(msg, GREEN if ok else RED)
                            if ok:
                                state = STATE_MODE_SELECT
                        elif fp:
                            set_status("File not found!", RED)
                        input_active = True
                    elif ev.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                        input_active = True
                    elif ev.key == pygame.K_v and (ev.mod & pygame.KMOD_CTRL):
                        clip = try_clipboard_paste()
                        if clip:
                            input_text += clip
                            input_active = True

                # ---- MODE SELECT state keys ----
                elif state == STATE_MODE_SELECT:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_1:
                        nav_mode = NAV_MODE_MANUAL
                    elif ev.key == pygame.K_2:
                        nav_mode = NAV_MODE_ASTAR
                    elif ev.key == pygame.K_RETURN:
                        state = STATE_NODE_MARKING
                        mode_name = "Manual Waypoints" if nav_mode == NAV_MODE_MANUAL else "D* Lite Adaptive"
                        set_status(f"Mode: {mode_name} — Place nodes now", GREEN)

                # ---- NODE MARKING state keys ----
                elif state == STATE_NODE_MARKING:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_d:
                        # Mark last placed node as destination
                        if node_ctr > 0:
                            last = node_ctr - 1
                            if last in nodes:
                                if dest_id is not None and dest_id in node_types:
                                    node_types[dest_id] = "normal"
                                dest_id = last
                                node_types[last] = "dest"
                                set_status(f"Node {last} → DESTINATION", RED)
                    elif ev.key == pygame.K_g:
                        show_grid = not show_grid
                        set_status(f"Grid: {'ON' if show_grid else 'OFF'}", CYAN)
                    elif ev.key == pygame.K_e:
                        edge_snap = not edge_snap
                        set_status(f"Edge Snap: {'ON' if edge_snap else 'OFF'}", CYAN)
                    elif ev.key == pygame.K_s:
                        # Validate & start
                        if len(nodes) < 2:
                            set_status("Need at least 2 nodes!", RED)
                        elif source_id is None:
                            set_status("No source node (first node placed is source)", RED)
                        elif dest_id is None:
                            set_status("No destination! Place a node, then press D", RED)
                        elif source_id == dest_id:
                            set_status("Source and destination are the same node!", RED)
                        else:
                            # Save nodes.json
                            try:
                                sd = {
                                    "nodes": {str(k): list(v) for k, v in nodes.items()},
                                    "source": source_id,
                                    "destination": dest_id,
                                    "types": {str(k): v for k, v in node_types.items()},
                                    "nav_mode": nav_mode,
                                }
                                sp = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "nodes.json")
                                with open(sp, "w") as f:
                                    json.dump(sd, f, indent=2)
                            except Exception:
                                pass

                            result = start_navigation()
                            if result is False:
                                pass  # status set inside
                            else:
                                drone = result
                                mode_label = "MANUAL" if nav_mode == NAV_MODE_MANUAL else "D* LITE"
                                set_status(f"Navigation started ({mode_label})", GREEN)

                # ---- NAVIGATING state keys ----
                elif state == STATE_NAVIGATING:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_m:
                        accurate_mode = not accurate_mode
                        set_status(f"Mode: {'ACCURATE' if accurate_mode else 'FAST'}", YELLOW)
                    elif ev.key == pygame.K_t:
                        show_true_pos = not show_true_pos
                        set_status(f"Debug overlay: {'ON — showing true position' if show_true_pos else 'OFF'}",
                                   GREEN if show_true_pos else GRAY)
                    elif ev.key == pygame.K_v:
                        # Cycle vision degradation mode
                        new_mode = vision_degradation.cycle()
                        vm_col = GREEN if new_mode == "CLEAR" else YELLOW if new_mode == "NOISY" else ORANGE if new_mode == "BLUR" else RED
                        set_status(f"Vision Mode: {new_mode}", vm_col)
                        set_banner(f"VISION MODE: {new_mode}", vm_col)
                        decision_explainer.add(f"Vision mode → {new_mode}", vm_col)
                        add_log(f"Vision: {new_mode}", vm_col)
                    elif ev.key == pygame.K_f:
                        # Toggle fullscreen
                        is_fullscreen = not is_fullscreen
                        if is_fullscreen:
                            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
                            scr_w, scr_h = screen.get_size()
                        else:
                            scr_w = max(MIN_WINDOW_W, img_w + 400)
                            scr_h = max(MIN_WINDOW_H, img_h + HUD_HEIGHT)
                            screen = pygame.display.set_mode((scr_w, scr_h))
                        ui_scale = max(0.6, min(scr_w / BASE_WIDTH, scr_h / BASE_HEIGHT))
                        set_status(f"{'Fullscreen' if is_fullscreen else 'Windowed'}", CYAN)
                    elif ev.key == pygame.K_l:
                        # Save UI layout
                        if panel_mgr.save(layout_path):
                            set_status("UI layout saved", GREEN)
                        else:
                            set_status("Layout save failed", RED)

                # ---- MISSION COMPLETE state keys ----
                elif state == STATE_MISSION_COMPLETE:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_r:
                        state = STATE_REPLAY
                        replay_system.reset()
                        replay_done = False

                # ---- POST REPLAY state keys ----
                elif state == STATE_POST_REPLAY:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_r:
                        # Replay again
                        state = STATE_REPLAY
                        replay_system.reset()
                        replay_done = False

                # ---- REPLAY state keys (ADVANCED CONTROLS) ----
                elif state == STATE_REPLAY:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        replay_system.toggle_pause()
                    elif ev.key == pygame.K_LEFT:
                        replay_system.seek(-30)  # rewind ~30 points
                    elif ev.key == pygame.K_RIGHT:
                        replay_system.seek(30)   # forward ~30 points
                    elif ev.key == pygame.K_0:
                        replay_system.set_speed(0.25)
                    elif ev.key == pygame.K_1:
                        replay_system.set_speed(0.5)
                    elif ev.key == pygame.K_2:
                        replay_system.set_speed(1)
                    elif ev.key == pygame.K_3:
                        replay_system.set_speed(2)

            # Text input (for loading path)
            elif ev.type == pygame.TEXTINPUT if hasattr(pygame, "TEXTINPUT") else -1:
                if state == STATE_LOADING:
                    input_text += ev.text
                    input_active = True

            # Mouse
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if state == STATE_LOADING:
                    input_active = True
                elif state == STATE_NODE_MARKING:
                    mx, my = ev.pos
                    if ev.button == 1 and my < img_h:
                        pos = (mx, my)
                        if edge_snap:
                            pos = snap_to_edge(pos)
                        nid = node_ctr
                        nodes[nid] = pos
                        if node_ctr == 0:
                            node_types[nid] = "source"
                            source_id = nid
                            set_status(f"Node {nid} placed (SOURCE)", GREEN)
                        else:
                            node_types[nid] = "normal"
                            set_status(f"Node {nid} placed", WHITE)
                        node_ctr += 1
                    elif ev.button == 3 and node_ctr > 0:
                        last = node_ctr - 1
                        if last in nodes:
                            if last == source_id:
                                source_id = None
                            if last == dest_id:
                                dest_id = None
                            del nodes[last]
                            if last in node_types:
                                del node_types[last]
                            node_ctr -= 1
                            set_status(f"Node {last} removed", ORANGE)
                elif state == STATE_NAVIGATING and ev.button == 1:
                    # Panel drag start
                    panel_mgr.handle_mouse_down(*ev.pos)
                elif state == STATE_REPLAY and ev.button == 1:
                    # Timeline seek on click
                    mx, my = ev.pos
                    tl_y = scr_h - 50
                    tl_x = 40
                    tl_w = scr_w - 80
                    if tl_y - 5 <= my <= tl_y + 25 and tl_x <= mx <= tl_x + tl_w:
                        progress = (mx - tl_x) / max(tl_w, 1)
                        replay_system.seek_to(clamp(progress, 0.0, 1.0))

            elif ev.type == pygame.MOUSEBUTTONUP:
                panel_mgr.handle_mouse_up()

            elif ev.type == pygame.MOUSEMOTION:
                if state == STATE_NAVIGATING:
                    panel_mgr.handle_mouse_move(*ev.pos, scr_w, scr_h)

        # ---- State logic ----

        if state == STATE_LOADING:
            if time.time() - load_t0 > 30:
                set_status("TIMEOUT: No image loaded. Exiting.", RED)
                draw_loading()
                pygame.display.flip()
                pygame.time.wait(2000)
                running = False
                continue
            draw_loading()

        elif state == STATE_NODE_MARKING:
            draw_node_marking()

        elif state == STATE_NAVIGATING:
            # Navigation logic
            if drone and path and path_idx < len(path):
                tid = path[path_idx]
                tpos = nodes[tid]

                if sub_state == SUB_MOVING:
                    # 1. Move drone toward target using ESTIMATED position for direction
                    move_speed = localizer.get_speed() if localizer else DRONE_SPEED

                    # Failure prediction: reduce speed in low-density regions
                    if failure_predictor and failure_predictor.warning_active:
                        move_speed *= 0.7

                    drone.move_toward_estimated(tpos, move_speed)

                    # 2. Run visual localization (updates estimated position)
                    if localizer:
                        localizer.localize(drone.true_pos, dt)
                        drone.update_estimated(localizer.estimated_pos)

                    drone.conf_trail.append(localizer.confidence if localizer else confidence)
                    update_reveal()

                    # --- Intelligence System Updates (every frame during movement) ---
                    loc_conf = localizer.confidence if localizer else confidence

                    # Heatmap: update local cells
                    if heatmap:
                        heatmap.update(drone.pos, loc_conf)

                    # Learning memory: tick decay
                    learning_memory.tick()

                    # Auto vision mode: update based on confidence
                    vision_degradation.update_auto(loc_conf, dt)

                    # Failure prediction: check ahead (requires both low density AND low confidence)
                    if failure_predictor and drone:
                        was_warning = failure_predictor.warning_active
                        failure_predictor.check_ahead(drone.pos, drone.heading, loc_conf, dt)
                        if failure_predictor.warning_active and not was_warning:
                            decision_explainer.add("\u26a0 Low feature density ahead", RED)

                    # Quick recovery: nudge toward target if stuck in low confidence > 2s
                    if localizer and localizer.loc_mode != LOC_NORMAL and localizer.lost_timer > 2.0:
                        tdx_r = tpos[0] - drone.est_x
                        tdy_r = tpos[1] - drone.est_y
                        tdist_r = math.sqrt(tdx_r * tdx_r + tdy_r * tdy_r)
                        if tdist_r > 1.0:
                            nudge = 8.0 * dt  # small forward bias
                            nx = drone.est_x + (tdx_r / tdist_r) * nudge
                            ny = drone.est_y + (tdy_r / tdist_r) * nudge
                            localizer.estimated_pos = (nx, ny)
                            drone.update_estimated((nx, ny))

                    # Decision explainer: localization mode changes
                    if localizer:
                        if localizer.loc_mode == LOC_SEARCHING:
                            if localizer.frame_counter % 30 == 0:  # avoid spam
                                decision_explainer.add("Low confidence \u2192 expanding search", YELLOW)
                        elif localizer.loc_mode == LOC_LOST:
                            if localizer.frame_counter % 30 == 0:
                                decision_explainer.add("Position lost \u2192 wide search active", RED)
                        # ORB/SIFT switching explanation
                        if localizer.matcher_used == "SIFT" and localizer.frame_counter % 20 == 0:
                            decision_explainer.add("ORB weak \u2192 switching to SIFT", YELLOW)

                    # 3. Arrival check based on ESTIMATED position
                    est_dist = euclidean_dist(drone.pos, tpos)
                    if est_dist < ARRIVAL_THRESHOLD:
                        sub_state = SUB_MATCHING
                        match_retries = 0
                        match_t0 = time.time()
                        add_log(f"ARRIVED Node {tid} \u2014 MATCHING", CYAN)
                        set_banner(f"NODE {tid} \u2014 PATTERN MATCHING", CYAN)
                        decision_explainer.add(f"Arrived at Node {tid} \u2192 matching", CYAN)

                    # 4. Check if localizer says we're lost too long \u2192 replan
                    if localizer and localizer.should_replan():
                        flash_t = 1.0
                        flash_col = YELLOW
                        sub_state = SUB_REPLANNING
                        replan_flash = 2.0
                        set_status("LOCALIZATION LOST \u2014 Adaptive Replanning...", RED)
                        set_banner("ADAPTIVE REPLANNING ACTIVE", RED)
                        add_log("LOCALIZATION LOST", RED)
                        decision_explainer.add("Localization lost \u2192 replanning", RED)
                        # Record replan event
                        replay_system.record_event(REPLAY_EVENT_REPLAN)

                elif sub_state == SUB_MATCHING:
                    # Use TRUE position for view extraction in matching
                    # (the camera sees from actual location)
                    try:
                        c, ms, sc = matcher.match(drone.true_pos, tid, tpos, accurate_mode)
                        confidence = c
                        match_scores = sc
                    except Exception:
                        confidence = 0.0
                        ms = "FAIL"

                    if ms == "CONFIRMED":
                        confirmed.append(tid)
                        path_idx += 1
                        add_log(f"Node {tid} CONFIRMED ({confidence*100:.0f}%)", GREEN)
                        set_banner(f"NODE {tid} CONFIRMED", GREEN)
                        decision_explainer.add(f"Node {tid} confirmed", GREEN)
                        # Target lock animation (green)
                        target_lock_active = True
                        target_lock_pos = tpos
                        target_lock_timer = target_lock_duration
                        target_lock_color = GREEN
                        # RANSAC explanation
                        if matcher and matcher.ransac_inlier_pct > 0.5:
                            decision_explainer.add("RANSAC verification passed", GREEN)
                        if path_idx >= len(path):
                            state = STATE_MISSION_COMPLETE
                            mission_t1 = time.time()
                            sub_state = SUB_COMPLETE
                            set_banner("MISSION COMPLETE", GREEN)
                            add_log("MISSION COMPLETE", GREEN)
                            decision_explainer.add("Mission complete!", GREEN)
                        else:
                            # --- Adaptive Path Refinement ---
                            # Skip intermediate node if next-next is closer/aligned
                            loc_conf_now = localizer.confidence if localizer else confidence
                            if loc_conf_now > 0.6 and path_idx + 1 < len(path):
                                next_nid = path[path_idx]
                                next_next_nid = path[path_idx + 1]
                                if next_nid in nodes and next_next_nid in nodes:
                                    d_direct = euclidean_dist(drone.pos, nodes[next_next_nid])
                                    d_via = (euclidean_dist(drone.pos, nodes[next_nid]) +
                                             euclidean_dist(nodes[next_nid], nodes[next_next_nid]))
                                    # Skip if going direct saves >20% distance
                                    if d_direct < d_via * 0.8 and d_direct < 200:
                                        confirmed.append(next_nid)
                                        path_idx += 1
                                        decision_explainer.add(f"Path refined: skipped Node {next_nid}", CYAN)
                                        add_log(f"SKIP Node {next_nid} (refined)", CYAN)
                            sub_state = SUB_MOVING

                    elif ms == "RETRY":
                        match_retries += 1
                        decision_explainer.add(f"Match failed \u2192 retry #{match_retries}", YELLOW)
                        # Record retry event
                        replay_system.record_event(REPLAY_EVENT_RETRY)
                        if match_retries >= MAX_RETRIES or (time.time() - match_t0) > MATCH_TIMEOUT:
                            skipped.append(tid)
                            blocked.add(tid)
                            flash_t = 1.0
                            flash_col = RED
                            sub_state = SUB_REPLANNING
                            replan_flash = 2.0
                            # Target lock animation (red = failure)
                            target_lock_active = True
                            target_lock_pos = tpos
                            target_lock_timer = target_lock_duration
                            target_lock_color = RED
                            # Record fail event
                            replay_system.record_event(REPLAY_EVENT_FAIL)
                            # Learning memory: record failure
                            learning_memory.record_failure(tpos)
                            decision_explainer.add(f"Learning: region penalized", ORANGE)
                        else:
                            sub_state = SUB_RETRYING

                    elif ms == "FAIL":
                        match_retries += 1
                        decision_explainer.add(f"Match FAIL at Node {tid}", RED)
                        # Record fail event
                        replay_system.record_event(REPLAY_EVENT_FAIL)
                        if match_retries >= MAX_RETRIES:
                            skipped.append(tid)
                            blocked.add(tid)
                            flash_t = 1.0
                            flash_col = RED
                            sub_state = SUB_REPLANNING
                            replan_flash = 2.0
                            # Learning memory: record failure
                            learning_memory.record_failure(tpos)
                            decision_explainer.add(f"Learning: region penalized", ORANGE)
                        else:
                            sub_state = SUB_RETRYING

                elif sub_state == SUB_RETRYING:
                    sub_state = SUB_MATCHING

                elif sub_state == SUB_REPLANNING:
                    # Reset localizer lost timer on replan
                    if localizer:
                        localizer.lost_timer = 0.0
                        localizer.loc_mode = LOC_SEARCHING

                    # Record replan event
                    replay_system.record_event(REPLAY_EVENT_REPLAN)
                    decision_explainer.add("Adaptive replanning triggered", YELLOW)

                    # --- ADAPTIVE REPLANNING (D* Lite-inspired) ---
                    if planner:
                        # Update blocked node in the planner
                        if tid in blocked:
                            planner.update_blocked(tid)

                        cur_node = path[max(0, path_idx - 1)]
                        np2 = planner.replan_from(cur_node, dest_id)

                        if np2 is None:
                            set_status("NO PATH AVAILABLE — Mission Failed", RED)
                            state = STATE_MISSION_COMPLETE
                            mission_t1 = time.time()
                            sub_state = SUB_COMPLETE
                            decision_explainer.add("No path available", RED)
                        else:
                            path[:] = np2
                            path_idx = 1
                            sub_state = SUB_MOVING
                            set_status("Adaptive replan successful", GREEN)
                            set_banner("ADAPTIVE REPLANNING ACTIVE", GREEN)
                            add_log(f"ADAPTIVE REPLAN #{planner.replan_count}", GREEN)
                            decision_explainer.add(f"Replan #{planner.replan_count} success", GREEN)
                            replan_flash = 1.5
                    else:
                        # Fallback: full rebuild (manual mode or no planner)
                        node_pos = {nid: nodes[nid] for nid in nodes}
                        g2 = build_knn_graph(node_pos, K_NEIGHBORS, blocked)
                        cur_node = path[max(0, path_idx - 1)]
                        np2 = a_star(g2, cur_node, dest_id, nodes)

                        if np2 is None:
                            set_status("NO PATH AVAILABLE — Mission Failed", RED)
                            state = STATE_MISSION_COMPLETE
                            mission_t1 = time.time()
                            sub_state = SUB_COMPLETE
                        else:
                            path[:] = np2
                            path_idx = 1
                            sub_state = SUB_MOVING
                            set_status("Replanned successfully", GREEN)
                            set_banner("ROUTE REPLANNED", GREEN)
                            add_log("REPLAN SUCCESS — new path", GREEN)
                            replan_flash = 0.5

                # Record replay every few frames
                if drone and len(drone.trail) % 4 == 0:
                    replay_system.record_point(
                        drone.ipos,
                        localizer.confidence if localizer else confidence
                    )

            draw_navigation()

        elif state == STATE_MODE_SELECT:
            draw_mode_select()

        elif state == STATE_MISSION_COMPLETE:
            draw_mission_complete()

        elif state == STATE_REPLAY:
            # Advance replay with smooth interpolation
            finished = replay_system.advance(dt)
            if finished:
                state = STATE_POST_REPLAY
            draw_replay()

        elif state == STATE_POST_REPLAY:
            draw_post_replay()

        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
