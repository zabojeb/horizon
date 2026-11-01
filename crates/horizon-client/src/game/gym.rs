//! `horizon-client --gym` (docs/gym.md): the game without a window, for training agents.
//!
//! The world runs on its server thread exactly as single player's does, and this window's
//! client joins it through memory; only the clock is gone. A `gym-step` holds the agent's
//! keys and buttons for its ticks, and each tick is the client's tick, then one server tick
//! granted through the [`Lockstep`] gate, then that tick's packets, read up to the server's
//! tick marker ([`gym::TICK_CHANNEL`]). Nothing waits for the wall clock: a world steps as fast
//! as it ticks, and a step is the same ticks however busy the machine is.
//!
//! Requests come over the control socket, one line each, one JSON reply each: `gym-info`,
//! `gym-palette`, `gym-config`, `gym-reset`, `gym-step`, `gym-obs`, `gym-command` and
//! `quit`. `tools/horizon_gym` is a Python client and a Gymnasium environment over them.
use super::{Game, Screen};
use horizon_client::{
    Result,
    gym::{self, Action, Lockstep, ObsConfig, VoxelBox},
};
use horizon_world::registry::BlockStateId;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{event::MouseButton, keyboard::KeyCode};

/// How long one server tick may take before a step gives up on it, and how long a reset may
/// wait for its world and the terrain around the player.
const TICK_TIMEOUT: Duration = Duration::from_secs(60);
const SETTLE_TIMEOUT: Duration = Duration::from_secs(300);
/// How long the server thread waits for a granted tick before feeding the chunk workers again.
const GRANT_POLL: Duration = Duration::from_millis(2);

/// The keys an action holds, as the client's movement reads them (`tick_remote_level`).
const MOVE_KEYS: [(fn(&Action) -> bool, KeyCode); 7] = [
    (|a| a.forward, KeyCode::KeyW),
    (|a| a.back, KeyCode::KeyS),
    (|a| a.left, KeyCode::KeyA),
    (|a| a.right, KeyCode::KeyD),
    (|a| a.jump, KeyCode::Space),
    (|a| a.sneak, KeyCode::ShiftLeft),
    (|a| a.sprint, KeyCode::ControlLeft),
];

/// What the gym keeps between requests (`Game::gym`).
#[derive(Default)]
pub(super) struct Gym {
    /// The gate with the open world's server thread.
    pub lockstep: Option<Arc<Lockstep>>,
    /// The last lockstep tick the server thread's marker named.
    pub marker: u64,
    /// The last sync marker the client read ([`gym::SYNC_CHANNEL`]), and the ones it asked for.
    pub synced: u64,
    syncs: u64,
    /// The `ClientTickEnd` packets this client sent to the open world.
    tick_ends: u64,
    config: ObsConfig,
    /// `Remote::chat_received` when the last observation was taken.
    chat_seen: u64,
    /// Worlds opened so far, the steps of this episode, and the episodes.
    worlds: u32,
    steps: u64,
    episode: u64,
    /// The world folder (in `saves/`) the gym created itself, which the next reload deletes.
    created: Option<PathBuf>,
}

/// `gym-reset`'s request.
#[derive(Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Reset {
    /// Open a fresh world with this seed (else the world goes on).
    seed: Option<i64>,
    /// Open a copy of this world folder, fresh for each reset.
    template: Option<PathBuf>,
    /// Open the world again even without a seed or a template (a new world with the game's
    /// `--seed`).
    reload: bool,
    /// Commands the server's console runs once the player is in the world (`/tp`, `/clear`,
    /// `/give`...), in order.
    commands: Vec<String>,
    /// The observation's contents from now on.
    config: Option<Value>,
    /// Ticks with no input after the commands, so that what they changed reaches the client.
    settle_ticks: Option<u32>,
}

/// The JSON of a block face.
fn face_name(face: [i32; 3]) -> &'static str {
    match face {
        [0, -1, 0] => "down",
        [0, 1, 0] => "up",
        [0, 0, -1] => "north",
        [0, 0, 1] => "south",
        [-1, 0, 0] => "west",
        _ => "east",
    }
}

fn round3(v: f64) -> f64 {
    (v * 1000.0).round() / 1000.0
}

/// Copy a world folder for a reset, without its `session.lock` (the copy is a world of its own).
fn copy_world(from: &Path, to: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let kind = entry.file_type()?;
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            copy_world(&entry.path(), &target)?;
        } else if kind.is_file() && entry.file_name() != "session.lock" {
            std::fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

impl Game<'_> {
    /// `--gym`: open the world and serve the agent's requests until it quits.
    pub(super) fn run_gym(&mut self) -> Result<()> {
        if self.control.is_none() {
            return Err(
                "--gym needs the control socket (--control-socket PATH); is No AI on?".into(),
            );
        }
        // The item and entity tables (`gym-palette`) and every world need them.
        self.join_world_rest()?;
        log::info!("Horizon: gym ready on the control socket");
        let result = self.serve_gym();
        if self.remote.is_some() || self.integrated.is_some() {
            // Leaving saves the world (kept, to look at) and stops its server thread.
            self.leave_server(Some("The gym closed".into()))?;
        }
        result
    }

    fn serve_gym(&mut self) -> Result<()> {
        loop {
            let request = match self.control.as_ref() {
                Some(control) => control.recv_timeout(Duration::from_millis(200)),
                None => return Ok(()),
            };
            let Some(request) = request else {
                // SIGTERM, SIGINT or SIGHUP: quit as `quit` does (the world is saved).
                if horizon_threads::stop_requests() > 0 {
                    return Ok(());
                }
                // Between requests the client still reads what reaches it.
                if self.remote.is_some() {
                    self.poll_remote()?;
                }
                continue;
            };
            let (word, body) = gym::split_request(&request.line);
            let reply = match word {
                "quit" | "gym-close" => {
                    let _ = request.reply.send(json!({ "ok": true }).to_string());
                    return Ok(());
                }
                "gym-info" => Ok(self.gym_info()),
                "gym-palette" => self.gym_palette(body),
                "gym-config" => self.gym_set_config(body),
                "gym-reset" => self.gym_reset(body),
                "gym-step" => self.gym_step(body),
                "gym-obs" => Ok(self.gym_observation()),
                "gym-command" => self.gym_console(body).map(|lines| json!({ "ok": true, "output": lines })),
                _ => Err(format!(
                    "unknown gym request {word}: gym-info, gym-palette, gym-config, gym-reset, gym-step, gym-obs, gym-command or quit"
                )
                .into()),
            };
            let reply = reply.unwrap_or_else(|e| json!({ "error": e.to_string() }));
            let _ = request.reply.send(reply.to_string());
        }
    }

    /// `gym-config CONFIG`: what observations carry from now on.
    fn gym_set_config(&mut self, body: &str) -> Result<Value> {
        let value: Value = if body.is_empty() {
            json!({})
        } else {
            serde_json::from_str(body).map_err(|e| format!("bad config: {e}"))?
        };
        self.gym.config = ObsConfig::parse(&value)?;
        Ok(json!({ "ok": true }))
    }

    fn gym_info(&self) -> Value {
        json!({
            "protocol": gym::PROTOCOL,
            "version": horizon_client::crash::version(),
            "block_states": self.pack.assets.registry.len(),
            "voxel_order": "yzx",
            "max_ticks": gym::MAX_TICKS,
            "config": {
                "voxels": self.gym.config.voxels,
                "entities": self.gym.config.entities,
                "entity_radius": self.gym.config.entity_radius,
                "inventory": self.gym.config.inventory,
                "chat": self.gym.config.chat,
            },
        })
    }

    /// `gym-palette blocks|items|entities`: the names behind the ids an observation carries.
    fn gym_palette(&self, kind: &str) -> Result<Value> {
        Ok(match kind {
            "blocks" => {
                let registry = &self.pack.assets.registry;
                let states: Vec<String> = (0..registry.len() as u32)
                    .map(|id| self.state_string(BlockStateId(id)))
                    .collect();
                json!({ "blocks": states })
            }
            "items" => {
                let items: Vec<&str> = self.items().registry_order.iter().map(|i| &**i).collect();
                json!({ "items": items })
            }
            "entities" => {
                let mut kinds: Vec<&String> = self.res.mob_types().keys().collect();
                kinds.sort();
                json!({ "entities": kinds })
            }
            other => {
                return Err(
                    format!("gym-palette takes blocks, items or entities, not {other:?}").into(),
                );
            }
        })
    }

    /// The server's console runs a command in the agent's level (permission level 4).
    fn gym_console(&mut self, command: &str) -> Result<Vec<String>> {
        let command = command.trim().trim_start_matches('/').to_owned();
        if command.is_empty() {
            return Err("gym-command needs a command".into());
        }
        let server = self
            .integrated
            .as_ref()
            .ok_or("no world is open; send gym-reset")?;
        server
            .call(move |g| g.owner_console_command(&command))
            .ok_or_else(|| "the server thread did not answer".into())
    }

    /// `gym-reset`: a new episode. The world goes on unless the request opens one (a seed, a
    /// template or `reload`); the player respawns if dead, the commands run, and the reply is
    /// the first observation.
    fn gym_reset(&mut self, body: &str) -> Result<Value> {
        let reset: Reset = if body.is_empty() {
            Reset::default()
        } else {
            serde_json::from_str(body).map_err(|e| format!("bad reset: {e}"))?
        };
        if let Some(config) = &reset.config {
            self.gym.config = ObsConfig::parse(config)?;
        }
        let reopen = reset.reload
            || reset.seed.is_some()
            || reset.template.is_some()
            || self.gym.lockstep.is_none();
        self.gym_release_input();
        if reopen {
            self.gym_open(reset.seed, reset.template.as_deref())?;
        }
        self.gym_settle()?;
        if self.screen == Screen::Dead {
            // The death screen's Respawn: the server's tick takes it, then its terrain loads.
            self.remote_respawn();
            let quiet = Action::default();
            for _ in 0..gym::MAX_TICKS {
                if self.screen != Screen::Dead {
                    break;
                }
                self.gym_tick(&quiet)?;
            }
            if self.screen == Screen::Dead {
                return Err("the player did not respawn".into());
            }
            self.gym_settle()?;
        }
        for command in &reset.commands {
            self.gym_console(command)?;
        }
        if !reset.commands.is_empty() {
            // What the commands sent reaches the client before anything else happens: a teleport
            // handled now or during the first quiet tick would start the episode from either of
            // two states. Then, as the teleport may land far away, its terrain as for a join.
            self.gym_sync()?;
            self.gym_settle()?;
        }
        let quiet = Action::default();
        for _ in 0..reset.settle_ticks.unwrap_or(1).min(gym::MAX_TICKS) {
            self.gym_tick(&quiet)?;
        }
        self.gym.episode += 1;
        self.gym.steps = 0;
        // The episode's chat starts now.
        self.gym.chat_seen = self.remote.as_ref().map_or(0, |r| r.chat_received);
        let mut obs = self.gym_observation();
        obs["episode"] = json!(self.gym.episode);
        Ok(obs)
    }

    /// Open a world for the gym on a new server thread with a new gate: a copy of `template`,
    /// a fresh world with `seed`, or (the first time) the `--world` folder.
    fn gym_open(&mut self, seed: Option<i64>, template: Option<&Path>) -> Result<()> {
        if self.remote.is_some() || self.integrated.is_some() {
            self.leave_server(Some("The gym opens another world".into()))?;
        }
        if let Some(gate) = self.gym.lockstep.take() {
            gate.close();
        }
        let saves = std::env::current_dir()?.join("saves");
        let first = self.gym.worlds == 0;
        let base = self.options.world.clone().unwrap_or_else(|| "gym".into());
        // The first world is `--world`; each later one a folder of the gym's own, which the
        // next reload deletes, so that a long run keeps one world on disk.
        let folder = if first && template.is_none() {
            base
        } else {
            format!("{base}-gym-{}", self.gym.worlds)
        };
        self.gym.worlds += 1;
        let path = saves.join(&folder);
        if let Some(previous) = self.gym.created.take()
            && previous != path
        {
            let _ = std::fs::remove_dir_all(&previous);
        }
        if !first || template.is_some() {
            // Never a folder the gym did not make: `-gym-` names are its own.
            let _ = std::fs::remove_dir_all(&path);
        }
        if !path.join("level.dat").is_file() {
            self.gym.created = Some(path.clone());
        }
        if let Some(template) = template {
            if !template.join("level.dat").is_file() {
                return Err(
                    format!("template {} is not a world folder", template.display()).into(),
                );
            }
            copy_world(template, &path)?;
        }
        let gate = Arc::new(Lockstep::default());
        gate.settle();
        self.gym.lockstep = Some(gate);
        self.gym.marker = 0;
        self.gym.synced = 0;
        self.gym.syncs = 0;
        self.gym.tick_ends = 0;
        self.agent_pointer = true;
        let (seed, creative) = (seed.unwrap_or(self.options.seed), self.options.creative);
        // Without a window this returns once the client joined.
        self.open_integrated(&folder, None, Some(seed), creative, None)?;
        if let Some(server) = &self.integrated {
            server.set_paused(false);
        }
        Ok(())
    }

    /// Wait until the player stands in the world with the terrain around it, while the server
    /// thread runs only its connections (`Lockstep::settle`): no world tick passes.
    fn gym_settle(&mut self) -> Result<()> {
        let gate = self.gym.lockstep.clone().ok_or("no world is open")?;
        gate.settle();
        let deadline = Instant::now() + SETTLE_TIMEOUT;
        loop {
            self.update_world();
            self.poll_stationary_client()?;
            if self.gym_ready() {
                return Ok(());
            }
            if self.integrated.as_ref().is_none_or(|s| s.stopped()) {
                return Err("the world's server thread stopped".into());
            }
            if Instant::now() > deadline {
                return Err("the player's terrain did not load".into());
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Read the connection until everything the server thread sent so far is handled: it sends
    /// a sync marker ([`gym::SYNC_CHANNEL`]) behind it, outside any tick, and the client reads up
    /// to that marker.
    fn gym_sync(&mut self) -> Result<()> {
        self.gym.syncs += 1;
        let target = self.gym.syncs;
        let server = self
            .integrated
            .as_ref()
            .ok_or("no world is open; send gym-reset")?;
        let sent = server
            .call(move |g| g.gym_send_to_owner(gym::SYNC_CHANNEL, target))
            .ok_or("the server thread did not answer")?;
        if !sent {
            return Err("the server has no connection for the player".into());
        }
        let deadline = Instant::now() + TICK_TIMEOUT;
        while self.gym.synced < target {
            self.poll_remote()?;
            if self.remote.is_none() {
                return Err("the client left the world".into());
            }
            if Instant::now() > deadline {
                return Err("the server thread's sync marker did not arrive".into());
            }
            if self.gym.synced < target {
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    /// The player is in the world, said so to the server, and its 3x3 chunks are lit.
    fn gym_ready(&self) -> bool {
        let Some(p) = &self.player else { return false };
        let sent = self.remote.as_ref().is_some_and(|r| r.loaded_sent);
        let (cx, cz) = (
            (p.position.x.floor() as i32) >> 4,
            (p.position.z.floor() as i32) >> 4,
        );
        sent && (-1..=1).all(|dz| (-1..=1).all(|dx| self.streaming.is_lit(cx + dx, cz + dz)))
    }

    /// Let go of every key and button the agent held.
    fn gym_release_input(&mut self) {
        for (_, key) in MOVE_KEYS {
            self.keys.remove(&key);
        }
        if self.left {
            let _ = self.play_mouse(MouseButton::Left, false);
        }
        if self.right {
            let _ = self.play_mouse(MouseButton::Right, false);
        }
    }

    /// Hold what the action holds: keys, buttons (pressed when they go down), the hotbar slot,
    /// and the view turned by its degrees.
    fn gym_hold(&mut self, a: &Action) -> Result<()> {
        if a.respawn && self.screen == Screen::Dead {
            self.remote_respawn();
        }
        for (held, key) in MOVE_KEYS {
            if held(a) {
                self.keys.insert(key);
            } else {
                self.keys.remove(&key);
            }
        }
        let playing = self.screen == Screen::Playing && self.player.is_some();
        if a.attack != self.left {
            self.play_mouse(MouseButton::Left, a.attack)?;
        }
        if a.use_item != self.right {
            self.play_mouse(MouseButton::Right, a.use_item)?;
        }
        if !playing {
            return Ok(());
        }
        if let Some(slot) = a.hotbar {
            self.selected = usize::from(slot - 1);
        }
        (self.yaw, self.pitch) = a.turn(self.yaw, self.pitch);
        if (a.drop || a.drop_stack) && !self.spectator() {
            self.remote_drop(a.drop_stack);
        }
        Ok(())
    }

    /// One lockstep tick: the client's tick (its input and movement, the packets they send,
    /// its `ClientTickEnd`), the server's tick, then every packet of that tick.
    fn gym_tick(&mut self, held: &Action) -> Result<f64> {
        let gate = self
            .gym
            .lockstep
            .clone()
            .ok_or("no world is open; send gym-reset")?;
        for (pressed, key) in MOVE_KEYS {
            if pressed(held) {
                self.keys.insert(key);
            }
        }
        self.tick()?;
        if self.remote.as_ref().is_some_and(|r| r.joined) {
            self.gym.tick_ends += 1;
        }
        let target = gate.grant(1, self.gym.tick_ends);
        let started = Instant::now();
        if !gate.wait_finished(target, TICK_TIMEOUT) {
            return Err(if self.integrated.as_ref().is_none_or(|s| s.stopped()) {
                "the world's server thread stopped".into()
            } else {
                format!(
                    "the server tick did not finish in {} s",
                    TICK_TIMEOUT.as_secs()
                )
                .into()
            });
        }
        let server_ms = started.elapsed().as_secs_f64() * 1000.0;
        // Every packet of that tick is ahead of its marker on the connection.
        let deadline = Instant::now() + TICK_TIMEOUT;
        while self.gym.marker < target {
            self.poll_remote()?;
            if self.remote.is_none() {
                return Err("the client left the world".into());
            }
            if Instant::now() > deadline {
                return Err("the server tick's packets did not arrive".into());
            }
            if self.gym.marker < target {
                std::thread::yield_now();
            }
        }
        self.update_world();
        Ok(server_ms)
    }

    /// `gym-step ACTION`: hold the action for its ticks, then observe.
    fn gym_step(&mut self, body: &str) -> Result<Value> {
        let action = Action::parse(body)?;
        if self.gym.lockstep.is_none() {
            return Err("no world is open; send gym-reset first".into());
        }
        let started = Instant::now();
        self.gym_hold(&action)?;
        let mut server_ms = 0.0;
        for _ in 0..action.ticks {
            server_ms += self.gym_tick(&action)?;
        }
        self.gym.steps += 1;
        let mut obs = self.gym_observation();
        obs["info"] = json!({
            "step_ms": round3(started.elapsed().as_secs_f64() * 1000.0),
            "server_ms": round3(server_ms),
            "ticks": action.ticks,
            "steps": self.gym.steps,
            "episode": self.gym.episode,
        });
        Ok(obs)
    }

    /// One server tick under the gym's gate (`run_integrated`'s loop): a granted tick runs the
    /// whole server tick and then sends its marker; while the client joins or rejoins, only the
    /// connections run (the command harness's join); otherwise the chunk workers are fed while
    /// the agent thinks.
    pub(super) fn gym_server_pass(&mut self, gate: &Lockstep) -> Result<()> {
        let joining = gate.settling() || !self.gym_owner_in_world();
        let wait = if joining { Duration::ZERO } else { GRANT_POLL };
        let Some(tick_ends) = gate.start(wait) else {
            if joining {
                self.tick_server_timed(true)?;
                std::thread::sleep(Duration::from_millis(1));
            } else {
                self.pump_chunks(Instant::now() + Duration::from_millis(1), false);
            }
            return Ok(());
        };
        self.gym_wait_for_client(tick_ends);
        let started = Instant::now();
        let result = self.tick_server_timed(false);
        let took = started.elapsed();
        // MinecraftServer.tickServer's tick times, for /tick query.
        self.clock.record(took);
        self.debug_charts.log_tick(started, took);
        let tick = gate.finished() + 1;
        self.gym_send_marker(tick);
        gate.finish();
        result
    }

    /// The window's player is in this server's world and its client said it is loaded.
    fn gym_owner_in_world(&self) -> bool {
        let owner = self.serving.as_ref().map(|s| s.owner);
        self.lan.as_ref().is_some_and(|h| {
            h.hub
                .guests
                .iter()
                .any(|g| Some(g.uuid) == owner && g.placed && g.loaded)
        })
    }

    /// Before a granted tick: the client's packets of its tick, up to its `ClientTickEnd`, have
    /// reached the hub's queue (the memory connection's thread forwards them).
    fn gym_wait_for_client(&self, tick_ends: u64) {
        let Some(host) = self.lan.as_ref() else {
            return;
        };
        let shared = &host.hub.lan.shared;
        let deadline = Instant::now() + Duration::from_secs(5);
        while shared.memory_tick_ends() < tick_ends {
            if Instant::now() > deadline {
                log::warn!(
                    "gym: the client's tick end {tick_ends} did not arrive (have {}); ticking anyway",
                    shared.memory_tick_ends()
                );
                return;
            }
            std::thread::yield_now();
        }
    }

    /// After a granted tick: its marker, behind every packet the tick sent.
    fn gym_send_marker(&mut self, tick: u64) {
        self.gym_send_to_owner(gym::TICK_CHANNEL, tick);
    }

    /// The server thread's side of a marker: `n` on `channel` to the window's own client, behind
    /// everything sent to it before. False when the player has no connection here.
    fn gym_send_to_owner(&mut self, channel: &str, n: u64) -> bool {
        let Some(owner) = self.serving.as_ref().map(|s| s.owner) else {
            return false;
        };
        let Some(host) = self.lan.as_mut() else {
            return false;
        };
        let Some(conn) = host
            .hub
            .guests
            .iter()
            .find(|g| g.uuid == owner)
            .map(|g| g.conn)
        else {
            return false;
        };
        host.hub.send_to(
            conn,
            &horizon_protocol::packets::play::Clientbound::CustomPayload(
                horizon_protocol::packets::configuration::CustomPayload {
                    channel: channel.into(),
                    data: gym::tick_payload(n),
                },
            ),
        );
        true
    }

    /// The observation: the player as its client knows it, the blocks around it, the nearest
    /// entities, the inventory and the new chat lines.
    fn gym_observation(&mut self) -> Value {
        let config = self.gym.config.clone();
        let dead = self.screen == Screen::Dead;
        let player = self.player.as_ref().map(|p| {
            json!({
                "pos": [p.position.x, p.position.y, p.position.z],
                "vel": [p.velocity.x, p.velocity.y, p.velocity.z],
                "yaw": self.yaw,
                "pitch": self.pitch,
                "on_ground": p.ground.on_ground,
                "flying": p.flying,
                "swimming": p.swimming,
                "sprinting": p.sprinting,
                "sneaking": self.keys.contains(&KeyCode::ShiftLeft),
                "health": p.health.health(),
                "absorption": p.health.absorption(),
                "food": p.health.food.food,
                "saturation": p.health.food.saturation,
                "air": p.air,
                "fire_ticks": p.fire_ticks,
                "fall_distance": p.fall_distance,
                "xp_level": self.experience.level,
                "xp_total": self.experience.total,
                "selected": self.selected + 1,
            })
        });
        let mut obs = json!({
            "tick": self.gym.marker,
            "dead": dead,
            "screen": self.screen_label(),
            "dimension": self.dim.id(),
            "time_of_day": self.world.day_time.rem_euclid(24000),
            "raining": self.world.weather.raining,
            "player": player,
            "target": self.gym_target(),
        });
        if config.voxels.iter().any(|&r| r > 0) {
            obs["voxels"] = self.gym_voxels(config.voxels);
        }
        if config.entities > 0 {
            obs["entities"] = self.gym_entities(config.entities, config.entity_radius);
        }
        if config.inventory {
            let slot = |s: &Option<horizon_client::items::ItemStack>| match s {
                Some(s) => json!([&*s.item, s.count]),
                None => Value::Null,
            };
            let mut slots: Vec<Value> = self.inventory.iter().map(slot).collect();
            slots.extend(self.armor.iter().map(slot));
            slots.push(slot(&self.offhand));
            obs["inventory"] = Value::Array(slots);
        }
        if config.chat {
            let (received, lines) = match &self.remote {
                Some(r) => {
                    let new = r.chat_received.saturating_sub(self.gym.chat_seen);
                    let from = r
                        .chat
                        .len()
                        .saturating_sub(usize::try_from(new).unwrap_or(usize::MAX));
                    let lines: Vec<&str> = r.chat[from..].iter().map(|(t, _)| t.as_str()).collect();
                    (r.chat_received, json!(lines))
                }
                None => (self.gym.chat_seen, json!([])),
            };
            self.gym.chat_seen = received;
            obs["chat"] = lines;
        }
        obs
    }

    /// The block or entity under the crosshair.
    fn gym_target(&self) -> Value {
        let Some(eye) = self.eye_position() else {
            return Value::Null;
        };
        let distance = |at: [f64; 3]| {
            let d: f64 = (0..3).map(|i| (at[i] - eye[i]).powi(2)).sum();
            d.sqrt()
        };
        let block = self.pick().map(|hit| {
            let state = self.world.block(hit.pos[0], hit.pos[1], hit.pos[2]);
            (
                distance(hit.point),
                json!({ "kind": "block", "pos": hit.pos, "face": face_name(hit.face),
                        "state": state.0, "distance": round3(distance(hit.point)) }),
            )
        });
        let entity = self.mob_hit().map(|(i, d)| {
            let m = &self.mobs[i];
            (
                d,
                json!({ "kind": "entity", "id": m.id, "type": m.kind.id, "health": m.health,
                        "distance": round3(d) }),
            )
        });
        match (block, entity) {
            (Some(b), Some(e)) => {
                if e.0 <= b.0 {
                    e.1
                } else {
                    b.1
                }
            }
            (Some(b), None) => b.1,
            (None, Some(e)) => e.1,
            (None, None) => Value::Null,
        }
    }

    /// The block states of the box around the player's feet (see `gym::VoxelBox`).
    fn gym_voxels(&self, radius: [u32; 3]) -> Value {
        let centre = self.player.as_ref().map_or([0, 0, 0], |p| {
            [
                p.position.x.floor() as i32,
                p.position.y.floor() as i32,
                p.position.z.floor() as i32,
            ]
        });
        let area = VoxelBox::around(centre, radius);
        let ids: Vec<u32> = area
            .positions()
            .map(|[x, y, z]| self.world.block(x, y, z).0)
            .collect();
        json!({
            "origin": area.origin,
            "shape": [area.size[1], area.size[2], area.size[0]],
            "dtype": "<u4",
            "data": gym::encode_ids(&ids),
        })
    }

    /// The nearest living mobs, dropped items and orbs, nearest first: type, offset from the
    /// player's feet, and what each is.
    fn gym_entities(&self, max: usize, radius: f64) -> Value {
        let Some(p) = &self.player else {
            return json!([]);
        };
        let me = [p.position.x, p.position.y, p.position.z];
        let mut found: Vec<(f64, Value)> = Vec::new();
        let mut push = |at: [f64; 3], v: Value| {
            let offset = [at[0] - me[0], at[1] - me[1], at[2] - me[2]];
            let d = offset.iter().map(|v| v * v).sum::<f64>().sqrt();
            if d <= radius {
                let mut v = v;
                v["offset"] = json!(offset.map(round3));
                v["distance"] = json!(round3(d));
                found.push((d, v));
            }
        };
        for m in self.mobs.iter().filter(|m| m.alive()) {
            let vel = [m.velocity.x, m.velocity.y, m.velocity.z].map(round3);
            push(
                [m.position.x, m.position.y, m.position.z],
                json!({ "id": m.id, "type": m.kind.id, "health": m.health, "vel": vel }),
            );
        }
        for item in &self.dropped {
            push(
                [item.position.x, item.position.y, item.position.z],
                json!({ "id": item.id, "type": "minecraft:item",
                        "item": [&*item.stack.item, item.stack.count] }),
            );
        }
        for orb in &self.orbs {
            push(
                [orb.position.x, orb.position.y, orb.position.z],
                json!({ "type": "minecraft:experience_orb", "value": orb.value }),
            );
        }
        found.sort_by(|a, b| a.0.total_cmp(&b.0));
        found.truncate(max);
        Value::Array(found.into_iter().map(|(_, v)| v).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faces_have_vanillas_names() {
        assert_eq!(face_name([0, 1, 0]), "up");
        assert_eq!(face_name([0, -1, 0]), "down");
        assert_eq!(face_name([1, 0, 0]), "east");
        assert_eq!(face_name([-1, 0, 0]), "west");
        assert_eq!(face_name([0, 0, 1]), "south");
        assert_eq!(face_name([0, 0, -1]), "north");
    }

    #[test]
    fn a_reset_request_is_checked() {
        let r: Reset = serde_json::from_str(
            r#"{"seed": 7, "commands": ["/clear"], "settle_ticks": 3, "config": {"voxels": [1, 1, 1]}}"#,
        )
        .unwrap();
        assert_eq!(r.seed, Some(7));
        assert_eq!(r.commands, ["/clear"]);
        assert_eq!(r.settle_ticks, Some(3));
        assert!(!r.reload && r.template.is_none());
        assert!(serde_json::from_str::<Reset>(r#"{"sead": 7}"#).is_err());
    }

    /// The keys held follow the action, one key each.
    #[test]
    fn movement_keys_follow_the_action() {
        let a = Action::parse(r#"{"forward": 1, "sneak": 1}"#).unwrap();
        let held: Vec<KeyCode> = MOVE_KEYS
            .iter()
            .filter(|(f, _)| f(&a))
            .map(|(_, k)| *k)
            .collect();
        assert_eq!(held, [KeyCode::KeyW, KeyCode::ShiftLeft]);
    }

    #[test]
    fn a_world_copy_leaves_its_lock_behind() {
        let root = std::env::temp_dir().join(format!("horizon-gym-copy-{}", std::process::id()));
        let (from, to) = (root.join("from"), root.join("to"));
        std::fs::create_dir_all(from.join("region")).unwrap();
        std::fs::write(from.join("level.dat"), b"level").unwrap();
        std::fs::write(from.join("session.lock"), b"lock").unwrap();
        std::fs::write(from.join("region/r.0.0.mca"), b"region").unwrap();
        copy_world(&from, &to).unwrap();
        assert_eq!(std::fs::read(to.join("level.dat")).unwrap(), b"level");
        assert_eq!(
            std::fs::read(to.join("region/r.0.0.mca")).unwrap(),
            b"region"
        );
        assert!(!to.join("session.lock").exists());
        std::fs::remove_dir_all(root).unwrap();
    }
}
