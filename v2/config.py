"""
Centralized configuration for Intelligent-Traffic-RL v2 (3-Junction Curved Corridor).

All network geometry, Indian Left-Hand Traffic (LHT) parameters, signal safety
timings, and simulation defaults are defined here so no module uses magic numbers.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
V2_ROOT = Path(__file__).resolve().parent
NETWORK_DIR = V2_ROOT / "network"
TRAFFIC_DIR = V2_ROOT / "traffic"
MODELS_DIR = V2_ROOT / "models"
REPORTS_DIR = V2_ROOT / "reports"
REPLAYS_DIR = V2_ROOT / "replays"
RESULTS_DIR = V2_ROOT / "results"
BASELINES_RESULTS_DIR = RESULTS_DIR / "baselines"
TRAINING_RESULTS_DIR = RESULTS_DIR / "training"
FINAL_RESULTS_DIR = RESULTS_DIR / "final"

NET_FILE = NETWORK_DIR / "corridor.net.xml"
NOD_FILE = NETWORK_DIR / "corridor.nod.xml"
EDG_FILE = NETWORK_DIR / "corridor.edg.xml"
CON_FILE = NETWORK_DIR / "corridor.con.xml"
NETCCFG_FILE = NETWORK_DIR / "corridor.netccfg"

# ---------------------------------------------------------------------------
# Road Network Geometry (3-Junction Indian LHT Corridor)
# ---------------------------------------------------------------------------
LEFTHAND = True

# Junction center coordinates (meters)
# J1 --(~315m curved)-- J2 --(~335m curved)-- J3
JUNCTION_COORDS = {
    "J1": (0.0, 0.0),
    "J2": (300.0, 35.0),
    "J3": (620.0, 0.0),
}

# External boundary node coordinates (meters) — all arms 220m - 270m
BOUNDARY_COORDS = {
    "W1": (-260.0, -25.0),
    "N1": (-15.0, 230.0),
    "S1": (15.0, -230.0),
    "N2": (315.0, 265.0),
    "S2": (285.0, -195.0),
    "N3": (605.0, 230.0),
    "S3": (635.0, -230.0),
    "E3": (880.0, 25.0),
}

# Lane & speed configuration
CORRIDOR_VEHICLE_LANES = 2          # 2 motorized lanes per direction on main E-W corridor
CROSS_STREET_VEHICLE_LANES = 2      # 2 motorized lanes per direction on N-S cross streets
VEHICLE_LANE_WIDTH_M = 3.5          # Standard urban arterial lane width (m)
SIDEWALK_WIDTH_M = 2.0              # Pedestrian sidewalk width (m)
CORRIDOR_SPEED_MPS = 13.89          # 50 km/h
CROSS_STREET_SPEED_MPS = 11.11      # 40 km/h
CURVE_SAMPLE_POINTS = 11            # Spline sample points for smooth curved edges
CARRIAGEWAY_OFFSET_M = 4.2          # Half-separation between opposing centerlines (spreadType="center")

# ---------------------------------------------------------------------------
# Signal Safety Rules & Baseline Controller Settings (Phase 3)
# ---------------------------------------------------------------------------
MIN_GREEN_S = 10
YELLOW_S = 3
ALL_RED_S = 2
MAX_GREEN_S = 60
PED_CLEARANCE_S = 5

# Fixed-Time baseline green durations (matching v1: 25s NS / 20s EW)
FIXED_TIME_NS_GREEN_S = 25
FIXED_TIME_EW_GREEN_S = 20

# SUMO Actuated baseline parameters (type="actuated")
ACTUATED_MAX_GAP_S = 3.0
ACTUATED_DETECTOR_GAP_S = 2.0

# Gridlock warning threshold (seconds a single vehicle waits continuously)
GRIDLOCK_WAIT_THRESHOLD_S = 300.0

# ---------------------------------------------------------------------------
# Shared DQN Controller & Training Settings (Phase 4)
# ---------------------------------------------------------------------------
DQN_STATE_DIM = 10                  # 4 queues + 4 waits + current_phase + phase_time
DQN_ACTION_DIM = 2                  # 0 = NS green, 1 = EW green
DQN_HIDDEN_DIMS = (128, 64)
DQN_DECISION_INTERVAL_S = 5         # Seconds between agent decisions

# State normalization caps
DQN_MAX_QUEUE_NORM = 25.0
DQN_MAX_WAIT_NORM = 300.0

# Reward weights: r = -(w_queue * sum_q + w_wait * sum_wait + w_switch * switched)
DQN_W_QUEUE = 1.0
DQN_W_WAIT = 0.05
DQN_W_SWITCH = 2.0

# Training hyperparameters (CPU PyTorch)
DQN_TOTAL_EPISODES = 100
DQN_TRAIN_DURATION_S = 900          # 15-min episodes for fast CPU training
DQN_BATCH_SIZE = 64
DQN_REPLAY_CAPACITY = 20000
DQN_GAMMA = 0.95
DQN_LR = 1e-3
DQN_EPSILON_START = 1.0
DQN_EPSILON_END = 0.05
DQN_TARGET_SYNC_EPISODES = 5
DQN_MIN_REPLAY_TO_TRAIN = 128

# ---------------------------------------------------------------------------
# Traffic & Pedestrian Generation (Phase 2)
# ---------------------------------------------------------------------------
STEP_LENGTH_S = 1.0
EPISODE_DURATION_S = 3600

# Separate seed ranges: training vs held-out evaluation
TRAIN_SEEDS = list(range(1, 51))
EVAL_SEEDS = list(range(1001, 1011))

# Three demand levels (vehicles per hour across the 3-junction corridor)
DEMAND_LEVELS_VPH = {
    "low": 720,
    "medium": 1330,
    "high": 1660,
}

# Indian mixed traffic vehicle type parameters (SUMO vType attributes)
VEHICLE_TYPES = {
    "motorcycle": {
        "length": 2.1,
        "width": 0.75,
        "minGap": 1.2,
        "maxSpeed": 15.28,  # ~55 km/h
        "accel": 3.2,
        "decel": 5.0,
        "sigma": 0.5,
        "guiShape": "motorcycle",
        "color": "59,130,246",
    },
    "car": {
        "length": 4.2,
        "width": 1.75,
        "minGap": 2.0,
        "maxSpeed": 13.89,  # 50 km/h
        "accel": 2.6,
        "decel": 4.5,
        "sigma": 0.4,
        "guiShape": "passenger",
        "color": "239,68,68",
    },
    "auto": {
        "length": 2.8,
        "width": 1.35,
        "minGap": 1.5,
        "maxSpeed": 11.11,  # 40 km/h
        "accel": 2.0,
        "decel": 4.0,
        "sigma": 0.55,
        "guiShape": "passenger/hatchback",
        "color": "245,158,11",
    },
    "bus": {
        "length": 10.5,
        "width": 2.5,
        "minGap": 2.8,
        "maxSpeed": 11.11,  # 40 km/h
        "accel": 1.2,
        "decel": 3.5,
        "sigma": 0.35,
        "guiShape": "bus",
        "color": "234,179,8",
    },
    "truck": {
        "length": 8.0,
        "width": 2.4,
        "minGap": 2.5,
        "maxSpeed": 11.11,  # 40 km/h
        "accel": 1.3,
        "decel": 3.5,
        "sigma": 0.35,
        "guiShape": "truck",
        "color": "14,165,233",
    },
}

# Configurable vehicle mix (must sum to 1.0)
VEHICLE_MIX = {
    "motorcycle": 0.45,
    "car": 0.25,
    "auto": 0.10,
    "bus": 0.10,
    "truck": 0.10,
}

# Boundary entry and exit edges of the 3-junction corridor
ENTRY_EDGES = [
    "W1_J1", "N1_J1", "S1_J1",
    "N2_J2", "S2_J2",
    "N3_J3", "S3_J3", "E3_J3",
]
EXIT_EDGES = [
    "J1_W1", "J1_N1", "J1_S1",
    "J2_N2", "J2_S2",
    "J3_N3", "J3_S3", "J3_E3",
]

# Pedestrians: small configurable crossing rate (pedestrians per hour)
PEDESTRIAN_RATE_VPH = 120
PEDESTRIAN_SPEED_MPS = 1.35
PEDESTRIAN_SPAWN_OFFSET_FROM_JUNCTION_M = 25.0
