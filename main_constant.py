# =============================================================================
# ROAD AND WORLD GEOMETRY
# Semua nilai ukuran dasar di file ini memakai world unit.
# Konversi cepat ke pixel: pixel = world unit * DEFAULT_SCALE.
# =============================================================================
LANE_COUNT = 3
LANE_WIDTH = 33  # 33 world unit ~= 49.5 px pada scale 1.5
ROAD_WIDTH = LANE_COUNT * LANE_WIDTH

FINISH_DISTANCE = 100  # garis finish selalu 100 unit setelah obstacle terjauh

# =============================================================================
# SCREEN AND CAMERA
# Mobil ditahan pada posisi layar tetap agar jalan terasa scrolling.
# =============================================================================
SCREEN_HEIGHT = 800  # tinggi viewport utama
CAR_STATIC_Y_POS = 150  # posisi Y mobil pada layar

DEFAULT_SCALE = 1.5  # 1 world unit = 1.5 pixel
FPS = 60

# =============================================================================
# VEHICLE DIMENSIONS
# 18 x 39 world unit ~= 27 x 58.5 pixel pada scale default.
# =============================================================================
CAR_WIDTH = 18
CAR_HEIGHT = 39

OBSTACLE_WIDTH = 18
OBSTACLE_HEIGHT = 39

USE_PNG = True

# =============================================================================
# VEHICLE DYNAMICS
# Semua speed internal memakai world_units/step.
# Konversi cepat: px/s = world_units/step * DEFAULT_SCALE * FPS.
# =============================================================================
CAR_OBSTACLE_SPEED = 2.1555555555555554  # ~= 194 px/s ~= 50 km/h
# CAR_OBSTACLE_SPEED = 2.379259259259259  
# Approx conversions (uses DEFAULT_SCALE=1.5 and FPS=60 -> px/s = world_units/step * 90)
# 55 km/h -> ~214.13 px/s -> ~2.379259259259259 world_units/step
# 60 km/h -> ~233.60 px/s -> ~2.5955555555555554 world_units/step
# 65 km/h -> ~253.07 px/s -> ~2.8129629629629627 world_units/step
# 70 km/h -> ~272.53 px/s -> ~3.0281481481481482 world_units/step
# 75 km/h -> ~292.00 px/s -> ~3.2444444444444446 world_units/step
OBSTACLE_SPEED = CAR_OBSTACLE_SPEED

# Kalibrasi saat ini memetakan 214 px/s ~= 55 km/h dan 292 px/s ~= 75 km/h.
CAR_MAX_SPEED = 3.2444444444444445  # ~= 292 px/s ~= 75 km/h ~= 20.83 m/s
CAR_MIN_SPEED = 2.3777777777777778  # ~= 214 px/s ~= 55 km/h ~= 15.28 m/s

# Setiap keputusan berlaku selama satu DECISION_INTERVAL penuh.
# Action 3, 4, 5 menaikkan speed sebesar SPEED_UP km/h per interval.
# Action 0, 1, 2 menurunkan speed sebesar SPEED_DOWN km/h per interval.
SPEED_UP = 2
SPEED_DOWN = -3

# =============================================================================
# CURRICULUM AND TRAINING CONTROL
# =============================================================================
DECISION_INTERVAL = 10  # 1 keputusan dipertahankan selama 10 simulation step
TURNING_ANGLE = 5  # perubahan steering target per keputusan kiri/kanan

TRAIN_MULTIPLIER = 5
MEMORY_SIZE = 150000

# =============================================================================
# DQN TRAINING HYPERPARAMETERS
# =============================================================================
LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 10
DQN_HIDDEN_SIZES = (128, 128, 64)
GRAD_CLIP_MAX_NORM = 1.0

KEYONE_MULTIPLIER = 50

CONSECUTIVE_SAVE_BEST = 3
CONSECUTIVE_STAGE_REQ = 1
ALLSTAGE_CONSECUTIVE_REQ = 1

INDEPENDENT_BASED = True
SUCCESS_BASED_REQ = 1
INDEPENDENT_COUNT_REQ = 10000
ValidationTesterMode = False

# =============================================================================
# RANDOM VISUALIZATION OBSTACLES
# =============================================================================
startRandom = 400
gapRandom = 125
maxRandom = 50
START_RANDOM = startRandom
GAP_RANDOM = gapRandom
MAX_RANDOM = maxRandom

# =============================================================================
# EPSILON MANAGEMENT
# Non-final stage dan final stage dapat memakai batas epsilon berbeda.
# =============================================================================
TRAIN_MAX_EPSILON = 1.0  # epsilon awal training
TRAIN_MIN_EPSILON = 0.01  # batas bawah stage non-final
TRAIN_FINAL_MIN_EPSILON = 0.01  # batas bawah final stage saat SSC masih 0
TRAIN_FINAL_MIN_EPSILON_SSC = 0.01  # batas bawah final stage saat SSC > 0
EPSILON_DECAY = 0.995  # decay per episode

CONSECUTIVE_EPSILON_RECOVERY = (
    10000  # episode di epsilon minimum sebelum recovery dipaksa
)
CONSECUTIVE_EPSILON_RECOVERY_SSC = (
    10000  # episode di floor epsilon final-stage SSC sebelum recovery
)
AMOUNT_EPSILON_RECOVERY = 0  # nilai epsilon saat recovery aktif
ENABLE_EPSILON_RECOVERY = False  # False = perilaku DQN standar

NEW_STAGE_EPSILON = 0  # 0 berarti pertahankan epsilon saat pindah stage
# =============================================================================
# SENSOR CONFIGURATION
# Urutan sensor untuk environment dan neural network:
# [R2, R1, F, L1, L2, SR, SL]
# =============================================================================
SENSOR_F = 100
SENSOR_L1 = 100
SENSOR_R1 = 100
SENSOR_L2 = 80
SENSOR_R2 = 80
SENSOR_SL = 40
SENSOR_SR = 40

SENSOR_ANGLE_F = 0
SENSOR_ANGLE_L1 = 15
SENSOR_ANGLE_R1 = -15
SENSOR_ANGLE_L2 = 40
SENSOR_ANGLE_R2 = -40
SENSOR_ANGLE_SL = 110
SENSOR_ANGLE_SR = -110

SENSOR_ANGLES = [
    SENSOR_ANGLE_R2,
    SENSOR_ANGLE_R1,
    SENSOR_ANGLE_F,
    SENSOR_ANGLE_L1,
    SENSOR_ANGLE_L2,
    SENSOR_ANGLE_SR,
    SENSOR_ANGLE_SL,
]


# =============================================================================
# REWARD SYSTEM
# Reward diarahkan ke lane tengah, heading lurus, dan keputusan fast-straight
# saat depan aman. Penalty utama datang dari collision dan jarak warning sensor.
# =============================================================================
OBSTACLE_WARNING_DISTANCE_FRONT = 60
OBSTACLE_WARNING_DISTANCE_SIDES = 16.5  # jarak warning sensor samping
STRAIGHT_ANGLE_THRESHOLD = 10
LANE_CENTER_REWARD_WIDTH = 8  # lebar zona reward di sekitar pusat lane
SHOW_CENTERLANE_REWARD_INDICATOR = False
CENTERLANE_REWARD_INDICATOR_COLOR = (0, 0, 255, 50)

LEFT_LR_OFFSETX = -3  # offset reward-zone lane kiri
RIGHT_LR_OFFSETX = 4
CENTER_LR_OFFSETX = 0.5
LEFT_OBSTACLE_OFFSETX = 0.0
RIGHT_OBSTACLE_OFFSETX = 0
CENTER_OBSTACLE_OFFSETX = 0.5

REWARD_PROGRESS = 0
REWARD_LANE_CENTER_MAX = 0.020
REWARD_STRAIGHT_ANGLE = 0.020
REWARD_FAST_CLEAR = 0.020
REWARD_FINISH = 10


PENALTY_COLLISION = -5.0
PENALTY_TIMEOUT = 0.0
PENALTY_WARNING_DISTANCE_FRONT = -0.030
PENALTY_WARNING_DISTANCE_SIDES = -0.030
PENALTY_NOT_IN_CENTER = -0.020
PENALTY_SLOW_WHEN_CLEAR = -0.030

# =============================================================================
# MAIN CURRICULUM STAGES
# OBSTACLES berisi stage training utama.
# Nilai y yang lebih besar berarti obstacle diletakkan lebih jauh ke depan.
# Panduan cepat jarak antar obstacle:
# - 150 = longgar
# - 120 = rapat
# - 50  = sangat ketat
# =============================================================================
OBSTACLES = [
    
    # [
    #     {"lane": 1, "y":100},
    #     {"lane": 1, "y":500},
    # ]
    
    # Stage utama training.
    [
        {"lane": 1, "y": 250},
        {"lane": 0, "y": 400},
        {"lane": 2, "y": 400},
        {"lane": 2, "y": 550},
        {"lane": 1, "y": 550},
        {"lane": 0, "y": 700},
        {"lane": 2, "y": 700},
        {"lane": 0, "y": 850},
        {"lane": 1, "y": 850},
        {"lane": 2, "y": 1000},
        {"lane": 1, "y": 1000},
        {"lane": 0, "y": 1125},
        {"lane": 1, "y": 1125},
        {"lane": 2, "y": 1275},
        {"lane": 0, "y": 1275},
        {"lane": 2, "y": 1450},
        {"lane": 1, "y": 1450},
        {"lane": 2, "y": 1600},
        {"lane": 0, "y": 1600},
        {"lane": 0, "y": 1750},
        {"lane": 1, "y": 1750},
        {"lane": 2, "y": 1875},
        {"lane": 1, "y": 1875},
        {"lane": 0, "y": 2000},
        {"lane": 1, "y": 2000},
        {"lane": 0, "y": 2125},
        {"lane": 2, "y": 2225},
    ],
    # Curriculum version
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 0, "y": 400},
    #     {"lane": 2, "y": 400},  #
    # ],
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 0, "y": 400},
    #     {"lane": 2, "y": 400},  #
    #     {"lane": 2, "y": 550},
    #     {"lane": 1, "y": 550},
    #     {"lane": 0, "y": 700},
    #     {"lane": 2, "y": 700},
    #     {"lane": 0, "y": 850},
    #     {"lane": 1, "y": 850},  #
    # ],
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 0, "y": 400},
    #     {"lane": 2, "y": 400},  #
    #     {"lane": 2, "y": 550},
    #     {"lane": 1, "y": 550},
    #     {"lane": 0, "y": 700},
    #     {"lane": 2, "y": 700},
    #     {"lane": 0, "y": 850},
    #     {"lane": 1, "y": 850},  #
    #     {"lane": 2, "y": 1000},
    #     {"lane": 1, "y": 1000},
    #     {"lane": 0, "y": 1125},
    #     {"lane": 1, "y": 1125},
    #     {"lane": 2, "y": 1275},
    #     {"lane": 0, "y": 1275},
    #     {"lane": 2, "y": 1450},
    #     {"lane": 1, "y": 1450},  #
    # ],
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 0, "y": 400},
    #     {"lane": 2, "y": 400},  #
    #     {"lane": 2, "y": 550},
    #     {"lane": 1, "y": 550},
    #     {"lane": 0, "y": 700},
    #     {"lane": 2, "y": 700},
    #     {"lane": 0, "y": 850},
    #     {"lane": 1, "y": 850},  #
    #     {"lane": 2, "y": 1000},
    #     {"lane": 1, "y": 1000},
    #     {"lane": 0, "y": 1125},
    #     {"lane": 1, "y": 1125},
    #     {"lane": 2, "y": 1275},
    #     {"lane": 0, "y": 1275},
    #     {"lane": 2, "y": 1450},
    #     {"lane": 1, "y": 1450},  #
    #     {"lane": 2, "y": 1600},
    #     {"lane": 0, "y": 1600},
    #     {"lane": 0, "y": 1750},
    #     {"lane": 1, "y": 1750},
    #     {"lane": 2, "y": 1875},
    #     {"lane": 1, "y": 1875},  #
    # ],
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 0, "y": 400},
    #     {"lane": 2, "y": 400},  #
    #     {"lane": 2, "y": 550},
    #     {"lane": 1, "y": 550},
    #     {"lane": 0, "y": 700},
    #     {"lane": 2, "y": 700},
    #     {"lane": 0, "y": 850},
    #     {"lane": 1, "y": 850},  #
    #     {"lane": 2, "y": 1000},
    #     {"lane": 1, "y": 1000},
    #     {"lane": 0, "y": 1125},
    #     {"lane": 1, "y": 1125},
    #     {"lane": 2, "y": 1275},
    #     {"lane": 0, "y": 1275},
    #     {"lane": 2, "y": 1450},
    #     {"lane": 1, "y": 1450},  #
    #     {"lane": 2, "y": 1600},
    #     {"lane": 0, "y": 1600},
    #     {"lane": 0, "y": 1750},
    #     {"lane": 1, "y": 1750},
    #     {"lane": 2, "y": 1875},
    #     {"lane": 1, "y": 1875},  #
    #     {"lane": 0, "y": 2000},
    #     {"lane": 1, "y": 2000},
    #     {"lane": 0, "y": 2125},
    #     {"lane": 2, "y": 2225},  #
    # ],
]


# =============================================================================
# TESTER STAGES
# TEST_OBSTACLES dipakai untuk validasi model setelah training.
# Nomor di komentar kanan dipakai sebagai referensi stage tester.
# =============================================================================
TEST_OBSTACLES = [
    # Single obstacle dan pasangan obstacle awal.
    # [{"lane": 0, "y": 250}],  # 1
    # [{"lane": 1, "y": 250}],  # 2
    # [{"lane": 2, "y": 250}],  # 3
    # [{"lane": 0, "y": 250}, {"lane": 2, "y": 250}],  # 4
    # [{"lane": 1, "y": 250}, {"lane": 0, "y": 250}],  # 5
    # [{"lane": 1, "y": 250}, {"lane": 2, "y": 250}],  # 6
    # # Base obstacle mulai lane kiri.
    # [{"lane": 0, "y": 250}, {"lane": 0, "y": 375}],  # 7
    # [{"lane": 0, "y": 250}, {"lane": 1, "y": 375}],  # 8
    # [{"lane": 0, "y": 250}, {"lane": 2, "y": 375}],  # 9
    # [{"lane": 0, "y": 250}, {"lane": 0, "y": 375}, {"lane": 2, "y": 375}],  # 10
    # [{"lane": 0, "y": 250}, {"lane": 1, "y": 375}, {"lane": 0, "y": 375}],  # 11
    # [{"lane": 0, "y": 250}, {"lane": 1, "y": 375}, {"lane": 2, "y": 375}],  # 12
    # # Base obstacle mulai lane tengah.
    # [{"lane": 1, "y": 250}, {"lane": 0, "y": 375}],  # 13
    # [{"lane": 1, "y": 250}, {"lane": 1, "y": 375}],  # 14
    # [{"lane": 1, "y": 250}, {"lane": 2, "y": 375}],  # 15
    # [{"lane": 1, "y": 250}, {"lane": 0, "y": 375}, {"lane": 2, "y": 375}],  # 16
    # [{"lane": 1, "y": 250}, {"lane": 1, "y": 375}, {"lane": 0, "y": 375}],  # 17
    # [{"lane": 1, "y": 250}, {"lane": 1, "y": 375}, {"lane": 2, "y": 375}],  # 18
    # # Base obstacle mulai lane kanan.
    # [{"lane": 2, "y": 250}, {"lane": 0, "y": 375}],  # 19
    # [{"lane": 2, "y": 250}, {"lane": 1, "y": 375}],  # 20
    # [{"lane": 2, "y": 250}, {"lane": 2, "y": 375}],  # 21
    # [{"lane": 2, "y": 250}, {"lane": 0, "y": 375}, {"lane": 2, "y": 375}],  # 22
    # [{"lane": 2, "y": 250}, {"lane": 1, "y": 375}, {"lane": 0, "y": 375}],  # 23
    # [{"lane": 2, "y": 250}, {"lane": 1, "y": 375}, {"lane": 2, "y": 375}],  # 24
    # # Base obstacle dua lane awal.
    # [{"lane": 0, "y": 250}, {"lane": 2, "y": 250}, {"lane": 0, "y": 375}],  # 25
    # [{"lane": 0, "y": 250}, {"lane": 2, "y": 250}, {"lane": 1, "y": 375}],  # 26
    # [{"lane": 0, "y": 250}, {"lane": 2, "y": 250}, {"lane": 2, "y": 375}],  # 27
    # [
    #     {"lane": 0, "y": 250},
    #     {"lane": 2, "y": 250},
    #     {"lane": 0, "y": 375},
    #     {"lane": 2, "y": 375},
    # ],  # 28
    # [
    #     {"lane": 0, "y": 250},
    #     {"lane": 2, "y": 250},
    #     {"lane": 1, "y": 375},
    #     {"lane": 0, "y": 375},
    # ],  # 29
    # [
    #     {"lane": 0, "y": 250},
    #     {"lane": 2, "y": 250},
    #     {"lane": 1, "y": 375},
    #     {"lane": 2, "y": 375},
    # ],  # 30
    # # Base obstacle kombinasi tengah-kiri.
    # [{"lane": 1, "y": 250}, {"lane": 0, "y": 250}, {"lane": 0, "y": 370}],  # 31
    # [{"lane": 1, "y": 250}, {"lane": 0, "y": 250}, {"lane": 1, "y": 370}],  # 32
    # [{"lane": 1, "y": 250}, {"lane": 0, "y": 250}, {"lane": 2, "y": 370}],  # 33
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 0, "y": 250},
    #     {"lane": 0, "y": 370},
    #     {"lane": 2, "y": 370},
    # ],  # 34
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 0, "y": 250},
    #     {"lane": 1, "y": 370},
    #     {"lane": 0, "y": 370},
    # ],  # 35
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 0, "y": 250},
    #     {"lane": 1, "y": 370},
    #     {"lane": 2, "y": 370},
    # ],  # 36
    # # Base obstacle kombinasi tengah-kanan.
    # [{"lane": 1, "y": 250}, {"lane": 2, "y": 250}, {"lane": 0, "y": 370}],  # 37
    # [{"lane": 1, "y": 250}, {"lane": 2, "y": 250}, {"lane": 1, "y": 370}],  # 38
    # [{"lane": 1, "y": 250}, {"lane": 2, "y": 250}, {"lane": 2, "y": 370}],  # 39
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 2, "y": 250},
    #     {"lane": 0, "y": 370},
    #     {"lane": 2, "y": 370},
    # ],  # 40
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 2, "y": 250},
    #     {"lane": 1, "y": 370},
    #     {"lane": 0, "y": 370},
    # ],  # 41
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 2, "y": 250},
    #     {"lane": 1, "y": 370},
    #     {"lane": 2, "y": 370},
    # ],  # 42
    # # Pola padat dengan ruang manuver sempit.
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 2, "y": 250},
    #     {"lane": 1, "y": 300},
    #     {"lane": 2, "y": 300},
    #     {"lane": 1, "y": 350},
    #     {"lane": 2, "y": 350},
    #     {"lane": 1, "y": 400},
    #     {"lane": 2, "y": 400},
    #     {"lane": 1, "y": 450},
    #     {"lane": 2, "y": 450},
    #     {"lane": 1, "y": 500},
    #     {"lane": 2, "y": 500},
    #     {"lane": 1, "y": 650},
    #     {"lane": 0, "y": 650},
    #     {"lane": 1, "y": 700},
    #     {"lane": 0, "y": 700},
    #     {"lane": 1, "y": 750},
    #     {"lane": 0, "y": 750},
    #     {"lane": 1, "y": 800},
    #     {"lane": 0, "y": 800},
    #     {"lane": 1, "y": 850},
    #     {"lane": 0, "y": 850},
    #     {"lane": 1, "y": 900},
    #     {"lane": 0, "y": 900},
    #     {"lane": 0, "y": 1050},
    #     {"lane": 2, "y": 1050},
    #     {"lane": 0, "y": 1100},
    #     {"lane": 2, "y": 1100},
    #     {"lane": 0, "y": 1150},
    #     {"lane": 2, "y": 1150},
    #     {"lane": 0, "y": 1200},
    #     {"lane": 2, "y": 1200},
    #     {"lane": 0, "y": 1250},
    #     {"lane": 2, "y": 1250},
    #     {"lane": 0, "y": 1300},
    #     {"lane": 2, "y": 1300},
    #     {"lane": 1, "y": 1420},
    # ],  # 43
    # # Pola zig-zag untuk forcing lane change berulang.
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 0, "y": 250},
    #     {"lane": 1, "y": 375},
    #     {"lane": 2, "y": 375},
    #     {"lane": 1, "y": 500},
    #     {"lane": 0, "y": 500},
    #     {"lane": 1, "y": 625},
    #     {"lane": 2, "y": 625},
    #     {"lane": 1, "y": 750},
    #     {"lane": 0, "y": 750},
    #     {"lane": 1, "y": 875},
    #     {"lane": 2, "y": 875},
    #     {"lane": 1, "y": 1000},
    #     {"lane": 0, "y": 1000},
    #     {"lane": 1, "y": 1125},
    #     {"lane": 2, "y": 1125},
    #     {"lane": 1, "y": 1250},
    #     {"lane": 0, "y": 1250},
    #     {"lane": 1, "y": 1375},
    #     {"lane": 2, "y": 1375},
    # ],  # 44
    # Stage panjang untuk validasi kompleks.
    [
        {"lane": 1, "y": 300},
        {"lane": 1, "y": 350},
        {"lane": 2, "y": 500},
        {"lane": 1, "y": 500},
        {"lane": 1, "y": 700},
        {"lane": 0, "y": 700},
        {"lane": 2, "y": 900},
        {"lane": 1, "y": 900},
        {"lane": 1, "y": 1050},
        {"lane": 0, "y": 1050},
        {"lane": 2, "y": 1200},
        {"lane": 1, "y": 1200},
        {"lane": 1, "y": 1325},
        {"lane": 0, "y": 1325},
        {"lane": 2, "y": 1470},
        {"lane": 0, "y": 1605},
        {"lane": 2, "y": 1605},
        {"lane": 0, "y": 1655},
        {"lane": 2, "y": 1655},
        {"lane": 0, "y": 1705},
        {"lane": 2, "y": 1705},
        {"lane": 1, "y": 1825},
        {"lane": 0, "y": 1955},
        {"lane": 2, "y": 1955},
        {"lane": 0, "y": 2075},
        {"lane": 1, "y": 2075},
        {"lane": 0, "y": 2125},
        {"lane": 1, "y": 2125},
        {"lane": 2, "y": 2245},
        {"lane": 1, "y": 2295},
        {"lane": 0, "y": 2415},
        {"lane": 1, "y": 2455},
        {"lane": 1, "y": 2575},
        {"lane": 0, "y": 2825},
        {"lane": 2, "y": 2825},
        {"lane": 0, "y": 3000},
        {"lane": 1, "y": 3050},
        {"lane": 2, "y": 3175},
        {"lane": 1, "y": 3225}
    ],  # 45
]
