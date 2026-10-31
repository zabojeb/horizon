# The gym: Horizon without a window, for training agents

`horizon-client --gym` runs the game with no window, no GPU and no sound, for reinforcement learning and
other programs that play many episodes. The world is vanilla 26.3's, run by the same server code single player,
LAN and dedicated servers use, and the agent plays as a player does, through the same client. The only thing
gone is the clock: the world steps when the agent steps it, as fast as it can tick, and every step is the same
number of ticks however busy the machine is.

**Status: experimental.** The lockstep gate, the actions, the observations and the Python client are tested
(`cargo test -p horizon-client --lib gym::`, `--bin horizon-client gym::`, `python3 -m unittest
tests.test_horizon_gym`, the last against a fake game). The first runs against a real world are still to be
measured on a machine with prepared resources.

## Quick start

```sh
cargo build --profile fast -p horizon-client --no-default-features   # no ALSA or udev needed
python3 -m tools.horizon_gym.bench --steps 2000                       # random actions, steps a second
```

The game needs prepared resources, as every run does ([README](../README.md#build-from-source)):
`.local/horizon-client/26.3` in this checkout, or `HORIZON_RESOURCES`. `HORIZON_CLIENT` names another build.

From Python (`tools/horizon_gym`; numpy and Gymnasium are optional, `raw=True` needs neither):

```python
from tools.horizon_gym import HorizonEnv, item_reward

env = HorizonEnv(game={'seed': 1234, 'render_distance': 4}, voxels=(4, 4, 4), frame_skip=2,
                 max_episode_steps=2000, reward=item_reward('minecraft:oak_log'),
                 reset_commands=['/clear', '/time set day'])
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step({'forward': 1, 'attack': 1, 'camera': [5.0, 0.0]})
env.close()
```

Several games in parallel are several processes, each with its own world: `gymnasium.vector.AsyncVectorEnv(
[lambda i=i: HorizonEnv(game={'seed': i}) for i in range(8)])`. Each `GameProcess` keeps to a scratch folder of
its own (its game folder, logs, `HORIZON_HOME`, `XDG_DATA_HOME`, `XDG_CONFIG_HOME`), so none of them touches a
player's worlds or settings.

## How a step runs

Single player already runs its world on a server thread that the window joins through memory
([server-levels.md](server-levels.md), `game/integrated.rs`). The gym keeps that and takes the clock out
(`game/gym.rs`, `horizon_client::gym`):

1. **The client's tick.** The agent's keys and buttons are held, its turn applied, and the window's `Game`
   ticks as a client of a server does: input, movement, the packets that tick sends, then
   `ServerboundClientTickEndPacket`.
2. **One server tick.** The client grants it through the `Lockstep` gate. The server thread first waits until
   the memory connection has handed that tick's `ClientTickEnd` to the hub (`Shared::memory_tick_ends`), so
   the tick sees every packet the client sent; then it runs the whole server tick, the levels, the hub and
   the players, as it would at 20 TPS.
3. **That tick's packets.** The server sends a `horizon:gym_tick` custom payload behind everything the tick
   sent, and the client reads its connection up to it.

While the agent thinks, nothing ticks: the server thread only feeds its chunk workers and runs requests from
the window (the reset commands). While the client joins or rejoins (`Lockstep::settle`), the server runs only
its connections, as the command harness's join does, so an episode starts with no tick spent.

What the gym leaves out because nothing is drawn: section meshes (`Streaming::mesh` is off), particles,
rain splashes and block animations (`tick_particles`), and sound. What it keeps: everything a player's client
does that the server hears, and everything the server does.

## Requests

The control socket (`--control-socket PATH`, the default endpoint otherwise) takes one line per request and
answers one JSON line, with `{"error": ...}` on failure (`horizon_client::control`).

| Request | Answer |
|---|---|
| `gym-info` | `protocol` (1), `version`, `block_states`, `voxel_order` (`yzx`), `max_ticks`, the observation `config` |
| `gym-palette blocks\|items\|entities` | the names behind the ids: block states by state id (`minecraft:oak_log[axis=y]`), items in registry order, entity types sorted |
| `gym-config {...}` | sets what observations carry (below) |
| `gym-reset {...}` | starts an episode; the first observation |
| `gym-step {...}` | holds an action for its ticks; the observation, with `info` (`step_ms`, `server_ms`, `ticks`, `steps`, `episode`) |
| `gym-obs` | the observation now, without a tick |
| `gym-command TEXT` | the server's console runs a command in the player's level (permission 4); its output lines |
| `quit` | saves the world and ends the game |

**Reset.** `{"seed": N}` opens a fresh world with that seed, `{"template": "PATH"}` a fresh copy of a saved world
folder (so every episode starts from the same terrain), `{"reload": true}` a fresh world with the game's
`--seed`; with none of them the world goes on and only the player is reset. A dead player respawns. Then
`commands` run in order on the server's console (`/clear`, `/tp @p 0 100 0`, `/give @p minecraft:stone_axe`,
`/gamerule ...`), the terrain around the player loads, and `settle_ticks` (1 by default) quiet ticks pass so that
what the commands changed reaches the client. `config` sets the observation as `gym-config` does. A world the
gym made is deleted by the next reset that opens one, so a long run keeps one world on disk (the last one stays,
to look at); a `--world` folder that existed before the gym started is never deleted.

**Action.** Every key is optional; buttons take `true`/`false` or a number (non-zero holds it):

| Key | Meaning |
|---|---|
| `forward`, `back`, `left`, `right`, `jump`, `sneak`, `sprint` | movement keys held for the step |
| `attack` | the attack button: going down attacks or starts digging, held it digs on (vanilla's `continueAttack`); release it for a step to hit again |
| `use` | the use button: going down uses or places; held, vanilla's right-click delay repeats it and eating, drinking or drawing goes on |
| `yaw`, `pitch` | degrees to turn the view by before the first tick (pitch stays within ±90) |
| `hotbar` | a slot, 1 to 9 |
| `drop`, `drop_stack` | drop one of the held item, or the stack |
| `respawn` | the death screen's Respawn |
| `ticks` | how many ticks the step lasts (frame skip), 1 to 200 |

**Observation.**

| Key | |
|---|---|
| `tick` | the lockstep ticks run since the world opened |
| `dead`, `screen`, `dimension`, `time_of_day`, `raining` | |
| `player` | `pos`, `vel`, `yaw`, `pitch`, `on_ground`, `flying`, `swimming`, `sprinting`, `sneaking`, `health`, `absorption`, `food`, `saturation`, `air`, `fire_ticks`, `fall_distance`, `xp_level`, `xp_total`, `selected` |
| `target` | the block (`pos`, `face`, `state`) or entity (`id`, `type`, `health`) under the crosshair, with its `distance`, or null |
| `voxels` | the block states of the box around the player's feet: `origin` (its lowest corner), `shape` `[y, z, x]`, `data` base64 of little-endian `u32` state ids, x fastest, then z, then y (`numpy.frombuffer(..., '<u4').reshape(shape)`) |
| `entities` | the nearest mobs, dropped items and orbs, nearest first: `type`, `offset` from the player, `distance`, and `id`, `health`, `vel` or `item` |
| `inventory` | 41 slots: the 36 of the inventory (0 to 8 the hotbar), the 4 armour slots (head first), the off hand; each `[item, count]` or null |
| `chat` | chat and system lines since the last observation (death messages among them) |

`gym-config` takes `voxels` (`[rx, ry, rz]`, blocks out from the feet on each axis, at most 16; `[0, 0, 0]` leaves
them out), `entities` (how many, at most 256), `entity_radius` (blocks), `inventory` and `chat` (on or off). The
default is `[4, 4, 4]` (9x9x9 blocks), 16 entities within 16 blocks, with the inventory and chat.

## Making it fast

- **Distances.** Terrain is most of a world's cost. `GameProcess` writes render distance 4 and simulation
  distance 5 into the game's `options.txt`; an agent that sees only voxels needs little more.
- **Frame skip.** `ticks` (or `frame_skip`) runs several ticks per request, which saves the request's round
  trip and the observation's encoding for each tick that is skipped.
- **Templates.** A pre-generated world (`template`) skips terrain generation at each reset, and makes every
  episode start the same.
- **Processes.** One game per core: the server tick runs on one thread, generation and light on the shared
  worker pool, so N games on N cores step independently.
- **The server build.** `--no-default-features` drops audio and controllers; nothing else changes.

## Determinism

A step is always one client tick and one server tick per `ticks`, in that order, and the world's own random
draws are vanilla's. What still varies between runs of one seed: when chunk generation on the worker threads
finishes (a chunk that arrives a tick later changes what spawns near it), and the wall clock where vanilla reads
it (keep-alives, chunk batch sizes). Episodes from a `template` with the same actions stay close; bit-exact
replays would need the server to wait for its workers each tick, which no option does yet.

## What is next

- **Pixels.** An observation drawn by the renderer (`--offscreen` with a GPU, read back as an image), in step
  with the world.
- **Several worlds in one process.** The `Resources` tables are shared already (`Arc`); one process holding many
  gyms would save their memory per world.
- **Lighter loading.** The gym loads every client resource, as a windowed client does. A client that never draws
  needs neither the block models nor the GUI atlases (the dedicated server already skips them).
- **Vanilla's memory connection.** A memory connection in vanilla has no read timeout
  (`Connection.configureInMemoryPipeline`); Horizon's integrated server keeps one of 30 s, which the gym turns
  off (`NetConfig::memory_timeout`). Check it against the bytecode and drop it for everyone if vanilla agrees.
