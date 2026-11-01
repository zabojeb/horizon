"""A Gymnasium environment over the game's gym mode (docs/gym.md).

`HorizonEnv` starts its own headless game (`GameProcess`) or uses one already running
(`socket=`). Its action is a dict of buttons (0 or 1), a `camera` turn in degrees
(`[yaw, pitch]`) and a `hotbar` slot (0 keeps the slot, 1 to 9 select one). Its observation
is a dict of numpy arrays: the block states around the player, its state, inventory and the
nearest entities (none with `entities=0`). `raw=True` returns the game's JSON instead, and
needs neither numpy nor Gymnasium.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .client import GymClient, decode_voxels
from .launch import GameProcess

try:  # Optional: the raw mode and the tests run without them.
    import numpy as np
except ImportError:  # pragma: no cover - depends on the machine
    np = None
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - depends on the machine
    gym = None
    spaces = None

#: The buttons of an action, in the order a flat vector would list them.
BUTTONS = ('forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack', 'use', 'drop')
#: The player's numbers in `obs['status']`.
STATUS = ('health', 'absorption', 'food', 'saturation', 'air', 'fire_ticks', 'fall_distance',
          'xp_level', 'on_ground', 'flying', 'swimming', 'sprinting', 'sneaking', 'dead')
#: An entity's row in `obs['entities']`.
ENTITY = ('type', 'dx', 'dy', 'dz', 'distance', 'health')

Reward = Callable[[Optional[dict], dict], float]


def game_action(action: Optional[dict], ticks: int = 1) -> dict:
    """The game's JSON action from this environment's action (or the game's own form)."""
    out: dict[str, Any] = {'ticks': int(ticks)}
    if not action:
        return out
    for key, value in action.items():
        if key == 'camera':
            yaw, pitch = (float(v) for v in value)
            out['yaw'], out['pitch'] = yaw, pitch
        elif key == 'hotbar':
            slot = int(value)
            if slot:
                out['hotbar'] = slot
        elif key in ('yaw', 'pitch'):
            out[key] = float(value)
        elif key == 'ticks':
            out['ticks'] = int(value)
        else:
            out[key] = 1 if int(value) else 0
    return out


def survival_reward(previous: Optional[dict], obs: dict) -> float:
    """A small reward for every step alive, and -1 for dying."""
    if obs.get('dead'):
        return -1.0
    return 0.01


def item_reward(item: str) -> Reward:
    """1 for each `item` gained since the last step (`minecraft:oak_log`...)."""
    def count(obs: Optional[dict]) -> int:
        if not obs:
            return 0
        return sum(slot[1] for slot in obs.get('inventory') or [] if slot and slot[0] == item)

    def reward(previous: Optional[dict], obs: dict) -> float:
        return float(max(0, count(obs) - count(previous))) if previous else 0.0
    return reward


_Base = gym.Env if gym is not None else object


class HorizonEnv(_Base):
    """One headless game as an environment. See the module."""

    metadata = {'render_modes': []}

    def __init__(self, *, socket: Optional[str] = None, game: Optional[dict] = None,
                 voxels: Sequence[int] = (4, 4, 4), entities: int = 16, entity_radius: float = 16.0,
                 frame_skip: int = 1, max_episode_steps: Optional[int] = None,
                 reset_commands: Sequence[str] = (), template: Optional[str] = None,
                 new_world_each_reset: bool = False, reward: Optional[Reward] = None,
                 raw: bool = False, client: Optional[GymClient] = None):
        self.raw = raw
        if not raw and np is None:
            raise ImportError('HorizonEnv needs numpy unless raw=True')
        self.config = {'voxels': [int(v) for v in voxels], 'entities': int(entities),
                       'entity_radius': float(entity_radius), 'inventory': True, 'chat': True}
        self.frame_skip = int(frame_skip)
        self.max_episode_steps = max_episode_steps
        self.reset_commands = list(reset_commands)
        self.template = template
        self.new_world_each_reset = new_world_each_reset
        self.reward_fn = reward or (lambda _previous, _obs: 0.0)
        self._blocks: Optional[list] = None
        self._last: Optional[dict] = None
        self._steps = 0
        self._opened = False
        self.process: Optional[GameProcess] = None
        if client is not None:
            self.client = client
        elif socket is not None:
            self.client = GymClient(socket)
        else:
            self.process = GameProcess(**(game or {}))
            self.client = self.process.start()
        try:
            self.info = self.client.info()
            self.items = {name: i + 1 for i, name in enumerate(self.client.palette('items'))}
            self.entity_types = {name: i + 1 for i, name in enumerate(self.client.palette('entities'))}
            if spaces is not None and not raw:
                self.action_space = self._action_space()
                self.observation_space = self._observation_space()
        except BaseException:
            # A game this environment started would outlive it: stop it before giving up (a
            # client the caller passed stays theirs).
            if client is None:
                self.close()
            raise

    # Spaces -------------------------------------------------------------------------------
    def _voxel_shape(self) -> tuple:
        rx, ry, rz = self.config['voxels']
        return (2 * ry + 1, 2 * rz + 1, 2 * rx + 1)

    def _action_space(self):
        parts = {name: spaces.Discrete(2) for name in BUTTONS}
        parts['camera'] = spaces.Box(-180.0, 180.0, shape=(2,), dtype=np.float32)
        parts['hotbar'] = spaces.Discrete(10)
        return spaces.Dict(parts)

    def _observation_space(self):
        n = self.config['entities']
        big = float(3.0e7)
        parts = {
            'voxels': spaces.Box(0, max(1, self.info['block_states'] - 1), shape=self._voxel_shape(),
                                 dtype=np.uint32),
            'position': spaces.Box(-big, big, shape=(3,), dtype=np.float64),
            'velocity': spaces.Box(-100.0, 100.0, shape=(3,), dtype=np.float32),
            'rotation': spaces.Box(-big, big, shape=(2,), dtype=np.float32),
            'status': spaces.Box(-big, big, shape=(len(STATUS),), dtype=np.float32),
            'selected': spaces.Discrete(10),
            'inventory': spaces.Box(0, max(len(self.items), 99) + 1, shape=(41, 2), dtype=np.int32),
        }
        # `entities=0` leaves the entities out (a MultiBinary of 0 is not a space).
        if n:
            parts['entities'] = spaces.Box(-big, big, shape=(n, len(ENTITY)), dtype=np.float32)
            parts['entity_mask'] = spaces.MultiBinary(n)
        return spaces.Dict(parts)

    # Episode ------------------------------------------------------------------------------
    def block_palette(self) -> list:
        """State strings by state id (`minecraft:oak_log[axis=y]`), fetched once."""
        if self._blocks is None:
            self._blocks = self.client.palette('blocks')
        return self._blocks

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if gym is not None and isinstance(self, gym.Env):
            super().reset(seed=seed)
        request: dict[str, Any] = {'config': self.config, 'commands': list(self.reset_commands)}
        if seed is not None:
            request['seed'] = int(seed)
        elif self.template and (self.new_world_each_reset or not self._opened):
            request['template'] = self.template
        elif self.new_world_each_reset and self._opened:
            request['reload'] = True
        request.update(options or {})
        obs = self.client.reset(**request)
        self._opened = True
        self._last = obs
        self._steps = 0
        return self.observation(obs), {'raw': obs}

    def step(self, action: Optional[dict]):
        obs = self.client.step(game_action(action, self.frame_skip))
        self._steps += 1
        reward = float(self.reward_fn(self._last, obs))
        self._last = obs
        terminated = bool(obs.get('dead'))
        truncated = self.max_episode_steps is not None and self._steps >= self.max_episode_steps
        return self.observation(obs), reward, terminated, truncated, {'raw': obs, **obs.get('info', {})}

    def close(self) -> None:
        if self.process is not None:
            self.process.stop(self.client)
            self.process = None
        else:
            self.client.close()

    # Observation --------------------------------------------------------------------------
    def observation(self, obs: dict):
        """The game's JSON observation as this environment's arrays (or as it is, raw)."""
        if self.raw:
            return obs
        player = obs.get('player') or {}
        shape = self._voxel_shape()
        # A writable copy: frombuffer's array is read-only, which some learners write into.
        voxels = (np.array(decode_voxels(obs['voxels'], np), dtype=np.uint32) if 'voxels' in obs
                  else np.zeros(shape, dtype=np.uint32))
        status = [float(player.get(key, 0) or 0) for key in STATUS[:-1]]
        status.append(1.0 if obs.get('dead') else 0.0)
        inventory = np.zeros((41, 2), dtype=np.int32)
        for i, slot in enumerate((obs.get('inventory') or [])[:41]):
            if slot:
                inventory[i] = (self.items.get(slot[0], 0), slot[1])
        out = {
            'voxels': voxels,
            'position': np.asarray(player.get('pos', (0.0, 0.0, 0.0)), dtype=np.float64),
            'velocity': np.asarray(player.get('vel', (0.0, 0.0, 0.0)), dtype=np.float32),
            'rotation': np.asarray((player.get('yaw', 0.0), player.get('pitch', 0.0)), dtype=np.float32),
            'status': np.asarray(status, dtype=np.float32),
            'selected': int(player.get('selected', 1)),
            'inventory': inventory,
        }
        n = self.config['entities']
        if n:
            entities = np.zeros((n, len(ENTITY)), dtype=np.float32)
            mask = np.zeros(n, dtype=np.int8)
            for i, e in enumerate((obs.get('entities') or [])[:n]):
                dx, dy, dz = e.get('offset', (0, 0, 0))
                entities[i] = (self.entity_types.get(e.get('type'), 0), dx, dy, dz, e.get('distance', 0),
                               e.get('health', 0) or 0)
                mask[i] = 1
            out['entities'], out['entity_mask'] = entities, mask
        return out
