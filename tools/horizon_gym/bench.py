"""Steps a headless game with random actions and reports how fast it steps (docs/gym.md).

    python3 -m tools.horizon_gym.bench --steps 2000 --seed 1234
    python3 -m tools.horizon_gym.bench --socket /path/gym.sock   # a game already running

It needs a `horizon-client` build and prepared resources (`find_game`, `find_resources`).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time

from .client import GymClient
from .env import BUTTONS
from .launch import GameProcess


def random_action(rng: random.Random) -> dict:
    action = {name: int(rng.random() < 0.3) for name in BUTTONS if name != 'drop'}
    action['yaw'] = rng.uniform(-15.0, 15.0)
    action['pitch'] = rng.uniform(-5.0, 5.0)
    return action


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--ticks', type=int, default=1, help='ticks per step (frame skip)')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--voxels', type=int, nargs=3, default=(4, 4, 4))
    parser.add_argument('--render-distance', type=int, default=4)
    parser.add_argument('--socket', help='a game already started with --gym')
    parser.add_argument('--game', help='the horizon-client to start')
    parser.add_argument('--resources', help='prepared resources')
    args = parser.parse_args()

    process = None
    started = time.monotonic()
    if args.socket:
        client = GymClient(args.socket)
    else:
        process = GameProcess(args.game, args.resources, seed=args.seed,
                              render_distance=args.render_distance)
        client = process.start()
    try:
        launch_s = time.monotonic() - started
        t = time.monotonic()
        client.reset(config={'voxels': list(args.voxels)})
        reset_s = time.monotonic() - t
        rng = random.Random(args.seed)
        step_ms, server_ms = [], []
        t = time.monotonic()
        for _ in range(args.steps):
            action = random_action(rng)
            action['ticks'] = args.ticks
            obs = client.step(action)
            info = obs.get('info', {})
            step_ms.append(info.get('step_ms', 0.0))
            server_ms.append(info.get('server_ms', 0.0))
            if obs.get('dead'):
                client.reset()
        elapsed = time.monotonic() - t
        print(json.dumps({
            'launch_s': round(launch_s, 2),
            'reset_s': round(reset_s, 2),
            'steps': args.steps,
            'ticks_per_step': args.ticks,
            'steps_per_s': round(args.steps / elapsed, 1),
            'ticks_per_s': round(args.steps * args.ticks / elapsed, 1),
            'realtime_factor': round(args.steps * args.ticks / elapsed / 20.0, 2),
            'step_ms_median': round(statistics.median(step_ms), 3),
            'server_ms_median': round(statistics.median(server_ms), 3),
        }, indent=2))
    finally:
        if process is not None:
            process.stop(client)
        else:
            client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
