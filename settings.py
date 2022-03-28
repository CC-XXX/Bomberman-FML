from collections import namedtuple
import pygame
from pygame.locals import *
import logging
from pathlib import Path

events = [
    'MOVED_LEFT',
    'MOVED_RIGHT',
    'MOVED_UP',
    'MOVED_DOWN',
    'WAITED',
    'INTERRUPTED',
    'INVALID_ACTION',

    'BOMB_DROPPED',
    'BOMB_EXPLODED',

    'CRATE_DESTROYED',
    'COIN_FOUND',
    'COIN_COLLECTED',

    'KILLED_OPPONENT',
    'KILLED_SELF',

    'GOT_KILLED',
    'OPPONENT_ELIMINATED',
    'SURVIVED_ROUND',
]
e = namedtuple('Events', events)(*range(len(events)))

# Game properties
COLS = 17
ROWS = 17
SCENARIOS = {
    "coin-heaven": {
        "CRATE_DENSITY": 0,
        "COIN_COUNT": 50
    },
    # This is the tournament game mode
    "classic": {
        "CRATE_DENSITY": 0.75,
        "COIN_COUNT": 9
    }
    # Feel free to add more game modes and properties
    # game is created in environment.py -> BombeRLeWorld -> build_arena()
}
MAX_AGENTS = 4

# Round properties
MAX_STEPS = 400

# GUI properties
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

ASSET_DIR = Path(__file__).parent / "assets"

AGENT_COLORS = ['blue', 'green', 'yellow', 'pink']

# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2  # = 1 of bomb explosion + N of lingering around

# Rules for agents
TIMEOUT = 0.5
TRAIN_TIMEOUT = float("inf")
REWARD_KILL = 5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_RETURN: 'WAIT',
    pygame.K_SPACE: 'BOMB',
}

# Logging levels
LOG_GAME = logging.INFO
LOG_AGENT_WRAPPER = logging.INFO
LOG_AGENT_CODE = logging.DEBUG
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN','WAIT','BOMB']

# find 4 perspectives from agent itself
DEFAULT_PERSPECTIVE_DISTANCE=3