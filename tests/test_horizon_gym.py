"""tools/horizon_gym against a fake game: the socket protocol, the voxel encoding, the
actions and observations, and the folders a game is started in (no game is started)."""

import base64
import json
import os
import socket
import struct
import tempfile
import threading
import unittest
from pathlib import Path

from tools.horizon_gym import client as gym_client
from tools.horizon_gym import env as gym_env
from tools.horizon_gym import launch as gym_launch


def voxels_json(origin, shape, ids):
    return {'origin': origin, 'shape': shape, 'dtype': '<u4',
            'data': base64.b64encode(struct.pack(f'<{len(ids)}I', *ids)).decode()}


OBS = {
    'tick': 3, 'dead': False, 'screen': 'playing', 'dimension': 'minecraft:overworld',
    'player': {'pos': [0.5, 64.0, 0.5], 'vel': [0.0, -0.08, 0.0], 'yaw': 90.0, 'pitch': 10.0,
               'on_ground': True, 'health': 20.0, 'food': 20, 'saturation': 5.0, 'air': 300,
               'selected': 2},
    'voxels': voxels_json([-1, 63, -1], [1, 1, 3], [1, 2, 3]),
    'entities': [{'id': 7, 'type': 'minecraft:pig', 'offset': [1.0, 0.0, 2.0], 'distance': 2.236,
                  'health': 10.0}],
    'inventory': [['minecraft:oak_log', 3]] + [None] * 40,
    'chat': [],
}


class FakeGame:
    """Answers gym requests as the game does: one JSON line per line, `error` on failure."""

    def __init__(self, tcp=False):
        self.requests = []
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        if tcp:
            self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listener.bind(('127.0.0.1', 0))
            self.path = root / 'gym.json'
            self.path.write_text(json.dumps({'transport': 'tcp', 'port': self.listener.getsockname()[1],
                                             'token': 'secret'}))
        else:
            self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.path = root / 'gym.sock'
            self.listener.bind(str(self.path))
        self.tcp = tcp
        self.listener.listen(1)
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def answer(self, line):
        word, _, body = line.partition(' ')
        self.requests.append((word, body))
        if word == 'gym-info':
            return {'protocol': 1, 'block_states': 10, 'voxel_order': 'yzx', 'max_ticks': 200}
        if word == 'gym-palette':
            return {body: {'items': ['minecraft:air', 'minecraft:oak_log'],
                           'entities': ['minecraft:cow', 'minecraft:pig'],
                           'blocks': ['minecraft:air', 'minecraft:stone']}[body]}
        if word in ('gym-reset', 'gym-obs'):
            return dict(OBS)
        if word == 'gym-step':
            action = json.loads(body)
            if action.get('ticks', 1) > 200:
                return {'error': 'bad action: ticks must be 1..=200'}
            obs = dict(OBS)
            obs['info'] = {'step_ms': 1.5, 'server_ms': 1.0, 'ticks': action.get('ticks', 1)}
            return obs
        if word == 'gym-command':
            return {'ok': True, 'output': [f'ran {body}']}
        if word == 'quit':
            return {'ok': True}
        return {'error': f'unknown gym request {word}'}

    def serve(self):
        conn, _ = self.listener.accept()
        with conn, conn.makefile('rwb') as f:
            if self.tcp:
                if f.readline().strip() != b'auth secret':
                    f.write(b'{"error":"bad or missing token"}\n')
                    f.flush()
                    return
                f.write(b'{"ok":true}\n')
                f.flush()
            for line in f:
                reply = self.answer(line.decode().strip())
                f.write(json.dumps(reply).encode() + b'\n')
                f.flush()

    def close(self):
        self.listener.close()
        self.dir.cleanup()


@unittest.skipUnless(hasattr(socket, 'AF_UNIX'), 'Unix sockets')
class ClientTests(unittest.TestCase):
    def setUp(self):
        self.game = FakeGame()
        self.client = gym_client.GymClient(self.game.path, timeout=10)

    def tearDown(self):
        self.client.close()
        self.game.close()

    def test_requests_are_one_line_each_with_json_bodies(self):
        self.assertEqual(self.client.info()['protocol'], gym_client.PROTOCOL)
        self.assertEqual(self.client.palette('items'), ['minecraft:air', 'minecraft:oak_log'])
        obs = self.client.step({'forward': 1, 'ticks': 4})
        self.assertEqual(obs['info']['ticks'], 4)
        self.assertEqual(self.client.command('/time set day'), ['ran /time set day'])
        words = [w for w, _ in self.game.requests]
        self.assertEqual(words, ['gym-info', 'gym-palette', 'gym-step', 'gym-command'])
        # Compact JSON, so that it stays on one line.
        self.assertEqual(self.game.requests[2][1], '{"forward":1,"ticks":4}')

    def test_errors_raise(self):
        with self.assertRaises(gym_client.GymError) as e:
            self.client.step({'ticks': 1000})
        self.assertIn('ticks', str(e.exception))
        with self.assertRaises(gym_client.GymError):
            self.client.request('gym-nothing')
        with self.assertRaises(ValueError):
            self.client.request('gym-command', 'a\nb')


class TcpTests(unittest.TestCase):
    def test_the_token_comes_first(self):
        game = FakeGame(tcp=True)
        try:
            with gym_client.GymClient(game.path, timeout=10) as client:
                self.assertEqual(client.info()['block_states'], 10)
            self.assertEqual(game.requests[0][0], 'gym-info')
        finally:
            game.close()


class VoxelTests(unittest.TestCase):
    def test_ids_decode_in_section_order(self):
        # A 3 x 2 x 2 box (x, y, z sizes): x fastest, then z, then y.
        ids = list(range(100, 112))
        v = voxels_json([10, 60, -5], [2, 2, 3], ids)
        flat = gym_client.decode_voxels(v)
        self.assertEqual(list(flat), ids)
        self.assertEqual(gym_client.voxel_at(v, flat, 10, 60, -5), 100)
        self.assertEqual(gym_client.voxel_at(v, flat, 11, 60, -5), 101)
        self.assertEqual(gym_client.voxel_at(v, flat, 10, 60, -4), 103)
        self.assertEqual(gym_client.voxel_at(v, flat, 12, 61, -4), 111)
        self.assertIsNone(gym_client.voxel_at(v, flat, 13, 60, -5))

    def test_large_ids_survive(self):
        v = voxels_json([0, 0, 0], [1, 1, 2], [0, 1 << 20])
        self.assertEqual(list(gym_client.decode_voxels(v)), [0, 1 << 20])

    def test_numpy_shape(self):
        try:
            import numpy
        except ImportError:
            self.skipTest('numpy')
        v = voxels_json([0, 0, 0], [2, 1, 3], [1, 2, 3, 4, 5, 6])
        a = gym_client.decode_voxels(v, numpy)
        self.assertEqual(a.shape, (2, 1, 3))
        self.assertEqual(int(a[1, 0, 2]), 6)


class ActionTests(unittest.TestCase):
    def test_env_actions_become_game_actions(self):
        action = {'forward': 1, 'jump': 0, 'attack': True, 'camera': [12.5, -3.0], 'hotbar': 0}
        self.assertEqual(gym_env.game_action(action, ticks=4),
                         {'ticks': 4, 'forward': 1, 'jump': 0, 'attack': 1, 'yaw': 12.5, 'pitch': -3.0})
        self.assertEqual(gym_env.game_action({'hotbar': 3}), {'ticks': 1, 'hotbar': 3})
        self.assertEqual(gym_env.game_action(None, ticks=2), {'ticks': 2})
        self.assertEqual(gym_env.game_action({'yaw': 1, 'ticks': 5}), {'ticks': 5, 'yaw': 1.0})

    def test_rewards(self):
        self.assertEqual(gym_env.survival_reward(None, {'dead': False}), 0.01)
        self.assertEqual(gym_env.survival_reward(None, {'dead': True}), -1.0)
        logs = gym_env.item_reward('minecraft:oak_log')
        before = {'inventory': [['minecraft:oak_log', 3], None]}
        after = {'inventory': [['minecraft:oak_log', 4], ['minecraft:oak_log', 1]]}
        self.assertEqual(logs(None, after), 0.0)
        self.assertEqual(logs(before, after), 2.0)
        self.assertEqual(logs(after, before), 0.0)


@unittest.skipUnless(hasattr(socket, 'AF_UNIX'), 'Unix sockets')
class EnvTests(unittest.TestCase):
    def setUp(self):
        self.game = FakeGame()

    def tearDown(self):
        self.game.close()

    def test_raw_episode(self):
        env = gym_env.HorizonEnv(socket=str(self.game.path), raw=True, frame_skip=2,
                                 max_episode_steps=2, reset_commands=['/clear'],
                                 reward=gym_env.survival_reward)
        obs, info = env.reset(seed=5)
        self.assertEqual(obs['tick'], 3)
        word, body = self.game.requests[-1]
        self.assertEqual(word, 'gym-reset')
        request = json.loads(body)
        self.assertEqual(request['seed'], 5)
        self.assertEqual(request['commands'], ['/clear'])
        self.assertEqual(request['config']['voxels'], [4, 4, 4])
        _, reward, terminated, truncated, info = env.step({'forward': 1})
        self.assertEqual((reward, terminated, truncated), (0.01, False, False))
        self.assertEqual(info['ticks'], 2)
        _, _, _, truncated, _ = env.step(None)
        self.assertTrue(truncated)
        env.close()

    def test_observations_as_arrays(self):
        try:
            import numpy
        except ImportError:
            self.skipTest('numpy')
        env = gym_env.HorizonEnv(socket=str(self.game.path), voxels=(1, 0, 0), entities=2)
        obs, _ = env.reset()
        self.assertEqual(obs['voxels'].shape, (1, 1, 3))
        self.assertEqual(obs['voxels'].tolist(), [[[1, 2, 3]]])
        self.assertEqual(obs['position'].tolist(), [0.5, 64.0, 0.5])
        self.assertEqual(obs['selected'], 2)
        # Items and entity types by palette index, from 1 (0 is none).
        self.assertEqual(obs['inventory'][0].tolist(), [2, 3])
        self.assertEqual(obs['inventory'][1].tolist(), [0, 0])
        self.assertEqual(obs['entities'][0][0], 2.0)
        self.assertEqual(obs['entity_mask'].tolist(), [1, 0])
        status = dict(zip(gym_env.STATUS, obs['status'].tolist()))
        self.assertEqual((status['health'], status['on_ground'], status['dead']), (20.0, 1.0, 0.0))
        if getattr(env, 'observation_space', None) is not None:
            self.assertTrue(env.observation_space.contains(obs))
        env.close()


class LaunchTests(unittest.TestCase):
    def test_a_game_keeps_to_its_scratch_folder(self):
        with tempfile.TemporaryDirectory() as scratch:
            game = gym_launch.GameProcess('/opt/horizon-client', '/res', world='w', seed=9,
                                          creative=True, render_distance=3, simulation_distance=4,
                                          scratch=scratch)
            argv = game.command()
            self.assertEqual(argv[:2], ['/opt/horizon-client', '--gym'])
            for flag, value in (('--world', 'w'), ('--seed', '9'), ('--render-distance', '3'),
                                ('--control-socket', str(game.socket)), ('--game-dir', str(game.game_dir))):
                self.assertEqual(argv[argv.index(flag) + 1], value)
            self.assertIn('--creative', argv)
            env = game.environment()
            for key in ('HORIZON_HOME', 'HORIZON_LAUNCHER_HOME', 'XDG_DATA_HOME', 'XDG_CONFIG_HOME',
                        'HORIZON_LAUNCHER_TRASH', 'HORIZON_LAUNCHER_CLIPBOARD', 'HORIZON_LAUNCHER_DESKTOP'):
                self.assertTrue(env[key].startswith(scratch), key)
            self.assertEqual(env['HORIZON_NO_AUDIO'], '1')
            self.assertEqual(env['HORIZON_LAUNCHER_NO_SELF_UPDATE'], '1')
            game.prepare()
            options = (game.game_dir / 'options.txt').read_text()
            self.assertIn('renderDistance:3', options)
            self.assertIn('simulationDistance:4', options)
            self.assertIn('pauseOnLostFocus:false', options)

    def test_missing_resources_are_named(self):
        with tempfile.TemporaryDirectory() as scratch:
            game = gym_launch.GameProcess('/opt/horizon-client', os.path.join(scratch, 'none'),
                                          scratch=scratch)
            with self.assertRaises(FileNotFoundError) as e:
                game.start()
            self.assertIn('prepared resources', str(e.exception))


if __name__ == '__main__':
    unittest.main()
