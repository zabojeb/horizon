"""Start a headless game for the gym in folders of its own (docs/gym.md).

Each game gets a fresh scratch folder holding its game folder (saves, `options.txt`), its home
(logs) and its data and config folders, so it never touches a player's own: `HORIZON_HOME`,
`HORIZON_LAUNCHER_HOME`, `XDG_DATA_HOME` and `XDG_CONFIG_HOME` all point into it. The rest of the
environment is `lab.game_env`'s, as for every tool that starts the game (no sound, no Discord, no
launcher self-update; AGENTS.md).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

from tools import lab

from .client import GymClient, GymError

REPO = Path(__file__).resolve().parents[2]
VERSION = '26.3'


def find_game(exe: Optional[os.PathLike | str] = None) -> Path:
    """The `horizon-client` to run: `exe`, `$HORIZON_CLIENT`, else this checkout's fast,
    release or debug build."""
    if exe:
        return Path(exe)
    if os.environ.get('HORIZON_CLIENT'):
        return Path(os.environ['HORIZON_CLIENT'])
    name = 'horizon-client.exe' if os.name == 'nt' else 'horizon-client'
    for profile in ('fast', 'release', 'debug'):
        candidate = REPO / 'target' / profile / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        'no horizon-client build: cargo build --profile fast -p horizon-client --no-default-features, '
        'or set HORIZON_CLIENT')


def find_resources(resources: Optional[os.PathLike | str] = None) -> Path:
    """The prepared resources: `resources`, `$HORIZON_RESOURCES`, else this checkout's
    `.local/horizon-client/26.3` (docs/parity-workflow.md)."""
    if resources:
        return Path(resources).resolve()
    if os.environ.get('HORIZON_RESOURCES'):
        return Path(os.environ['HORIZON_RESOURCES']).resolve()
    return REPO / '.local' / 'horizon-client' / VERSION


def options_txt(render_distance: int, simulation_distance: int) -> str:
    """The game's `options.txt` for a gym: short distances (less terrain to generate and
    send), no pause without focus."""
    return '\n'.join([
        f'renderDistance:{render_distance}',
        f'simulationDistance:{simulation_distance}',
        'pauseOnLostFocus:false',
        'onboardAccessibility:false',
        'tutorialStep:none',
        'joinedFirstServer:true',
        '',
    ])


class GameProcess:
    """One `horizon-client --gym` with its own scratch folder.

    `world` and `seed` name the first world (later ones come from `gym-reset`), `creative`
    opens it in creative mode, and the distances go to its `options.txt`.
    """

    def __init__(self, exe: Optional[os.PathLike | str] = None,
                 resources: Optional[os.PathLike | str] = None, *, world: str = 'gym',
                 seed: int = 0, creative: bool = False, render_distance: int = 4,
                 simulation_distance: int = 5, extra_args: Sequence[str] = (),
                 scratch: Optional[os.PathLike | str] = None, keep: bool = False,
                 start_timeout: float = 600.0):
        self.exe = find_game(exe)
        self.resources = find_resources(resources)
        self.world, self.seed, self.creative = world, seed, creative
        self.render_distance, self.simulation_distance = render_distance, simulation_distance
        self.extra_args = list(extra_args)
        self.keep = keep or scratch is not None
        self.scratch = Path(scratch) if scratch else Path(tempfile.mkdtemp(prefix='horizon-gym-'))
        self.start_timeout = start_timeout
        self.game_dir = self.scratch / 'game'
        self.socket = self.game_dir / ('gym.json' if os.name == 'nt' else 'gym.sock')
        self.log_path = self.scratch / 'game.log'
        self.process: Optional[subprocess.Popen] = None

    def command(self) -> list[str]:
        """The command line the game starts with."""
        args = [str(self.exe), '--gym', '--game-dir', str(self.game_dir), '--home',
                str(self.scratch / 'home'), '--control-socket', str(self.socket),
                '--resources', str(self.resources), '--world', self.world, '--seed', str(self.seed),
                '--render-distance', str(self.render_distance)]
        if self.creative:
            args.append('--creative')
        return args + self.extra_args

    def environment(self) -> dict[str, str]:
        """The game's environment: `lab.game_env`'s, with every folder it writes in the scratch
        folder."""
        env = lab.game_env(
            HORIZON_HOME=self.scratch / 'home',
            HORIZON_LAUNCHER_HOME=self.scratch / 'launcher',
            XDG_DATA_HOME=self.scratch / 'data',
            XDG_CONFIG_HOME=self.scratch / 'config',
            HORIZON_LAUNCHER_TRASH=self.scratch / 'trash',
            HORIZON_LAUNCHER_CLIPBOARD=self.scratch / 'clipboard',
            HORIZON_LAUNCHER_DESKTOP=self.scratch / 'desktop',
        )
        env.pop('HORIZON_CONTROL', None)
        return env

    def prepare(self) -> None:
        """The scratch folders and the game's `options.txt`."""
        for name in ('game', 'home', 'launcher', 'data', 'config'):
            (self.scratch / name).mkdir(parents=True, exist_ok=True)
        options = self.game_dir / 'options.txt'
        if not options.exists():
            options.write_text(options_txt(self.render_distance, self.simulation_distance))

    def start(self) -> GymClient:
        """Start the game and connect once its socket answers."""
        if not (self.resources / 'horizon-resources.json').is_file():
            raise FileNotFoundError(
                f'{self.resources} holds no prepared resources (horizon-prepare --prepare, '
                'or set HORIZON_RESOURCES)')
        self.prepare()
        log = open(self.log_path, 'wb')
        try:
            self.process = subprocess.Popen(self.command(), cwd=self.game_dir, env=self.environment(),
                                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        finally:
            log.close()
        deadline = time.monotonic() + self.start_timeout
        while True:
            if self.process.poll() is not None:
                raise GymError(f'the game exited ({self.process.returncode}) before its socket opened:\n'
                               + self.log_tail())
            if self.socket.exists():
                try:
                    return GymClient(self.socket)
                except OSError:
                    pass
            if time.monotonic() > deadline:
                self.stop()
                raise GymError('the game did not open its socket in time:\n' + self.log_tail())
            time.sleep(0.05)

    def log_tail(self, lines: int = 30) -> str:
        try:
            return '\n'.join(self.log_path.read_text(errors='replace').splitlines()[-lines:])
        except OSError:
            return ''

    def stop(self, client: Optional[GymClient] = None, timeout: float = 60.0) -> None:
        """Quit the game (it saves its world), then remove the scratch folder unless kept."""
        if client is not None:
            client.quit()
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.wait(timeout=timeout if client is not None else 0.1)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if not self.keep:
            shutil.rmtree(self.scratch, ignore_errors=True)
