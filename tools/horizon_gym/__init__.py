"""Horizon's headless training mode from Python (docs/gym.md).

- `GymClient`: the requests of a game started with `--gym`, over its control socket.
- `GameProcess`: start such a game in a scratch folder of its own.
- `HorizonEnv`: a Gymnasium environment over both (numpy and Gymnasium are optional: without
  them, `raw=True` gives the game's JSON).

Everything players run is Rust; this is a client for researchers, like the harness drivers.
"""

from .client import PROTOCOL, GymClient, GymError, decode_voxels, voxel_at
from .env import BUTTONS, ENTITY, STATUS, HorizonEnv, game_action, item_reward, survival_reward
from .launch import GameProcess, find_game, find_resources

__all__ = [
    'BUTTONS', 'ENTITY', 'PROTOCOL', 'STATUS', 'GameProcess', 'GymClient', 'GymError', 'HorizonEnv',
    'decode_voxels', 'find_game', 'find_resources', 'game_action', 'item_reward', 'survival_reward',
    'voxel_at',
]
