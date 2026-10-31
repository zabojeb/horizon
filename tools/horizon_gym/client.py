"""The game's gym requests over its control socket (docs/gym.md).

One request is one line, `WORD [JSON]`, and the game answers with one JSON line; an answer
with an `error` key raises `GymError`. The socket is a Unix socket on Linux and macOS, or, on
Windows (and for a path ending in `.json`), a localhost TCP port described by that JSON file,
whose `token` the first line must present (`horizon_client::control`).
"""

from __future__ import annotations

import array
import base64
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any, Optional

PROTOCOL = 1


class GymError(RuntimeError):
    """The game answered with an error, or went away."""


def uses_tcp(path: Path) -> bool:
    """The game's own rule (`control::uses_tcp`): TCP off Unix, or for a `.json` endpoint."""
    return os.name != 'posix' or path.suffix == '.json'


class GymClient:
    """A connection to one game started with `--gym`."""

    def __init__(self, path: os.PathLike | str, timeout: Optional[float] = None):
        self.path = Path(path)
        token = None
        if uses_tcp(self.path):
            endpoint = json.loads(self.path.read_text())
            token = endpoint['token']
            sock = socket.create_connection(('127.0.0.1', int(endpoint['port'])), timeout=timeout)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        else:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect(str(self.path))
        self._sock = sock
        self._file = sock.makefile('rwb')
        if token is not None:
            self._send(f'auth {token}')
            self._receive()

    def _send(self, line: str) -> None:
        if '\n' in line or '\r' in line:
            raise ValueError('a gym request is one line')
        self._file.write(line.encode('utf-8') + b'\n')
        self._file.flush()

    def _receive(self) -> Any:
        reply = self._file.readline()
        if not reply:
            raise GymError('the game closed the connection')
        value = json.loads(reply)
        if isinstance(value, dict) and 'error' in value:
            raise GymError(value['error'])
        return value

    def request(self, word: str, body: Any = None) -> Any:
        """Send `word` with `body` (JSON-encoded unless it is text) and return the answer."""
        if body is None:
            line = word
        elif isinstance(body, str):
            line = f'{word} {body}'
        else:
            line = f"{word} {json.dumps(body, separators=(',', ':'))}"
        self._send(line)
        return self._receive()

    # The requests, by name.
    def info(self) -> dict:
        return self.request('gym-info')

    def palette(self, kind: str) -> list:
        """`blocks` (state strings by state id), `items` (by registry id) or `entities`."""
        return self.request('gym-palette', kind)[kind]

    def config(self, **config: Any) -> dict:
        return self.request('gym-config', config)

    def reset(self, **request: Any) -> dict:
        return self.request('gym-reset', request)

    def step(self, action: Optional[dict] = None) -> dict:
        return self.request('gym-step', action or {})

    def observe(self) -> dict:
        return self.request('gym-obs')

    def command(self, text: str) -> list:
        return self.request('gym-command', text)['output']

    def quit(self) -> None:
        try:
            self.request('quit')
        except (GymError, OSError):
            pass
        self.close()

    def close(self) -> None:
        for closer in (self._file.close, self._sock.close):
            try:
                closer()
            except OSError:
                pass

    def __enter__(self) -> 'GymClient':
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


def decode_voxels(voxels: dict, numpy: Any = None):
    """An observation's `voxels` as block state ids: a numpy array of shape (y, z, x) when
    `numpy` is given, else a flat `array.array` in the same order (x fastest)."""
    raw = base64.b64decode(voxels['data'])
    if numpy is not None:
        return numpy.frombuffer(raw, dtype='<u4').reshape(tuple(voxels['shape']))
    code = 'I' if array.array('I').itemsize == 4 else 'L'
    ids = array.array(code)
    ids.frombytes(raw)
    if sys.byteorder == 'big':
        ids.byteswap()
    return ids


def voxel_at(voxels: dict, ids, x: int, y: int, z: int) -> Optional[int]:
    """The state id at world position x, y, z in a decoded flat box, or None outside it."""
    ox, oy, oz = voxels['origin']
    sy, sz, sx = voxels['shape']
    dx, dy, dz = x - ox, y - oy, z - oz
    if not (0 <= dx < sx and 0 <= dy < sy and 0 <= dz < sz):
        return None
    return ids[(dy * sz + dz) * sx + dx]
