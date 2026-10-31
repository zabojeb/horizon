//! The headless training mode (`horizon-client --gym`, docs/gym.md): what the game's gym loop
//! (`game/gym.rs`) and its Python client (`tools/horizon_gym`) agree on, kept here so it is
//! tested without a world.
//!
//! - [`Lockstep`]: the gate between the agent's client and the server thread. The server ticks
//!   the world only for ticks the client granted, and the client waits for them, so a step is
//!   always the same number of client and server ticks, with no wait for the wall clock.
//! - [`Action`]: what an agent holds for a step (keys, buttons, a turn of the view).
//! - [`ObsConfig`] and [`VoxelBox`]: which blocks and entities an observation carries, and
//!   [`encode_ids`], the compact form of the block states.

use base64::Engine as _;
use serde::{Deserialize, Deserializer};
use std::{
    sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError},
    time::{Duration, Instant},
};

/// The protocol revision `gym-info` reports; a client checks it before it steps.
pub const PROTOCOL: u32 = 1;

/// The custom payload the server thread sends its own client after each lockstep tick, with
/// the tick's number ([`tick_payload`]): every packet of that tick is ahead of it on the
/// connection, so the client has them all once it has handled this one.
pub const TICK_CHANNEL: &str = "horizon:gym_tick";

/// The most ticks one step may hold its action for.
pub const MAX_TICKS: u32 = 200;
/// The most blocks a voxel radius may reach on one axis (33 blocks across).
pub const MAX_VOXEL_RADIUS: u32 = 16;
/// The most entities an observation may list.
pub const MAX_ENTITIES: usize = 256;

/// The tick marker's data: the tick's number, 8 bytes big-endian.
pub fn tick_payload(tick: u64) -> Vec<u8> {
    tick.to_be_bytes().to_vec()
}
/// The tick a marker names; `None` for data that is not one.
pub fn read_tick_payload(data: &[u8]) -> Option<u64> {
    Some(u64::from_be_bytes(data.try_into().ok()?))
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Gate {
    /// Ticks the client allowed so far.
    granted: u64,
    /// Ticks the server started, and finished.
    started: u64,
    finished: u64,
    /// The `ClientTickEnd` packets the client had sent when it granted the last tick: the
    /// server waits for its connection to have forwarded as many before it ticks.
    tick_ends: u64,
    /// The client is (re)joining: the server runs only its connections, as the command
    /// harness's join does, until the next grant.
    settling: bool,
    closed: bool,
}

/// The lockstep gate (see the module). Both sides poll it with short timeouts, so neither
/// stops serving its own work (the server's tasks and chunk workers, the client's packets).
#[derive(Debug, Default)]
pub struct Lockstep {
    gate: Mutex<Gate>,
    changed: Condvar,
}
impl Lockstep {
    fn gate(&self) -> MutexGuard<'_, Gate> {
        self.gate.lock().unwrap_or_else(PoisonError::into_inner)
    }
    /// The client side: allow `n` more server ticks, having sent `tick_ends` `ClientTickEnd`
    /// packets in all. Returns the count of finished ticks to wait for.
    pub fn grant(&self, n: u64, tick_ends: u64) -> u64 {
        let mut gate = self.gate();
        gate.granted += n;
        gate.tick_ends = gate.tick_ends.max(tick_ends);
        gate.settling = false;
        let target = gate.granted;
        drop(gate);
        self.changed.notify_all();
        target
    }
    /// The client side: the client is joining or rejoining; the server serves its connection
    /// without ticking the world until the next [`Self::grant`].
    pub fn settle(&self) {
        self.gate().settling = true;
        self.changed.notify_all();
    }
    /// The server side: start a granted tick if one is waiting, waiting up to `timeout` for
    /// one. `Some(tick_ends)` is the `ClientTickEnd` count to wait for before ticking.
    pub fn start(&self, timeout: Duration) -> Option<u64> {
        let deadline = Instant::now() + timeout;
        let mut gate = self.gate();
        loop {
            if gate.closed {
                return None;
            }
            if gate.started < gate.granted {
                gate.started += 1;
                return Some(gate.tick_ends);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return None;
            }
            gate = self
                .changed
                .wait_timeout(gate, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
    /// The server side: the tick [`Self::start`] began is done. Returns its number (1 for
    /// the first).
    pub fn finish(&self) -> u64 {
        let mut gate = self.gate();
        gate.finished = (gate.finished + 1).min(gate.started);
        let tick = gate.finished;
        drop(gate);
        self.changed.notify_all();
        tick
    }
    /// The client side: wait until `target` ticks finished (`true`), up to `timeout`.
    pub fn wait_finished(&self, target: u64, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        let mut gate = self.gate();
        loop {
            if gate.finished >= target {
                return true;
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if gate.closed || left.is_zero() {
                return false;
            }
            gate = self
                .changed
                .wait_timeout(gate, left)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }
    /// Ticks finished so far.
    pub fn finished(&self) -> u64 {
        self.gate().finished
    }
    /// Whether the client is (re)joining ([`Self::settle`]).
    pub fn settling(&self) -> bool {
        self.gate().settling
    }
    /// Either side goes away: every wait returns.
    pub fn close(&self) {
        self.gate().closed = true;
        self.changed.notify_all();
    }
}

/// Closes the gate when dropped. The server thread holds one, so a server that stops, fails or
/// panics frees the agent waiting on it.
pub struct CloseOnDrop(pub Arc<Lockstep>);
impl Drop for CloseOnDrop {
    fn drop(&mut self) {
        self.0.close();
    }
}

/// A button or key in JSON: `true`/`false`, or a number (non-zero is held), as an agent's
/// numeric action vector sends it.
fn held<'de, D: Deserializer<'de>>(d: D) -> Result<bool, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Held {
        Flag(bool),
        Number(f64),
    }
    Ok(match Held::deserialize(d)? {
        Held::Flag(b) => b,
        Held::Number(n) => n != 0.0,
    })
}

/// What the agent holds for one step: its keys and buttons are down for the step's ticks
/// (and released at its end unless the next step holds them too), and the view turns by
/// `yaw` and `pitch` degrees before the first tick.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Action {
    #[serde(deserialize_with = "held")]
    pub forward: bool,
    #[serde(deserialize_with = "held")]
    pub back: bool,
    #[serde(deserialize_with = "held")]
    pub left: bool,
    #[serde(deserialize_with = "held")]
    pub right: bool,
    #[serde(deserialize_with = "held")]
    pub jump: bool,
    #[serde(deserialize_with = "held")]
    pub sneak: bool,
    #[serde(deserialize_with = "held")]
    pub sprint: bool,
    /// The attack button (left click): a press attacks or starts digging, holding it digs on.
    #[serde(deserialize_with = "held")]
    pub attack: bool,
    /// The use button (right click): a press uses or places, holding it repeats as vanilla's
    /// right-click delay allows, and keeps eating, drinking or drawing a bow.
    #[serde(rename = "use", deserialize_with = "held")]
    pub use_item: bool,
    /// Degrees to turn the view by: yaw to the right, pitch down (vanilla's signs).
    pub yaw: f32,
    pub pitch: f32,
    /// A hotbar slot, 1 to 9, to select before the first tick.
    pub hotbar: Option<u8>,
    /// Drop one of the held item (Q), or the whole stack (`drop_stack`), on the first tick.
    #[serde(deserialize_with = "held")]
    pub drop: bool,
    #[serde(deserialize_with = "held")]
    pub drop_stack: bool,
    /// Respawn from the death screen.
    #[serde(deserialize_with = "held")]
    pub respawn: bool,
    /// How many ticks the step lasts, 1 to [`MAX_TICKS`] (frame skip).
    pub ticks: u32,
}
impl Default for Action {
    fn default() -> Self {
        Self {
            forward: false,
            back: false,
            left: false,
            right: false,
            jump: false,
            sneak: false,
            sprint: false,
            attack: false,
            use_item: false,
            yaw: 0.0,
            pitch: 0.0,
            hotbar: None,
            drop: false,
            drop_stack: false,
            respawn: false,
            ticks: 1,
        }
    }
}
impl Action {
    /// Parse and check a step's action; empty text is the no-op action.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim();
        let action: Self = if text.is_empty() {
            Self::default()
        } else {
            serde_json::from_str(text).map_err(|e| format!("bad action: {e}"))?
        };
        if !action.yaw.is_finite() || !action.pitch.is_finite() {
            return Err("bad action: yaw and pitch must be finite".into());
        }
        if !(1..=MAX_TICKS).contains(&action.ticks) {
            return Err(format!("bad action: ticks must be 1..={MAX_TICKS}"));
        }
        if action.hotbar.is_some_and(|slot| !(1..=9).contains(&slot)) {
            return Err("bad action: hotbar is a slot 1..=9".into());
        }
        Ok(action)
    }
    /// `Entity.turn` with the action's degrees: the yaw wraps freely, the pitch stays within
    /// straight down and straight up.
    pub fn turn(&self, yaw: f32, pitch: f32) -> (f32, f32) {
        (yaw + self.yaw, (pitch + self.pitch).clamp(-90.0, 90.0))
    }
}

/// What an observation carries (`gym-config`, or the `config` of `gym-reset`).
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct ObsConfig {
    /// Blocks around the block the player's feet are in, on x, y (down and up) and z; 0 on
    /// every axis leaves the voxels out.
    pub voxels: [u32; 3],
    /// The nearest entities listed (mobs, players, items, orbs), at most this many.
    pub entities: usize,
    /// How far (blocks) entities are looked for.
    pub entity_radius: f64,
    /// The inventory's 41 slots.
    pub inventory: bool,
    /// Chat and system messages since the last observation.
    pub chat: bool,
}
impl Default for ObsConfig {
    fn default() -> Self {
        Self {
            voxels: [4, 4, 4],
            entities: 16,
            entity_radius: 16.0,
            inventory: true,
            chat: true,
        }
    }
}
impl ObsConfig {
    /// Parse and check an observation configuration.
    pub fn parse(value: &serde_json::Value) -> Result<Self, String> {
        let config: Self =
            serde_json::from_value(value.clone()).map_err(|e| format!("bad config: {e}"))?;
        if config.voxels.iter().any(|&r| r > MAX_VOXEL_RADIUS) {
            return Err(format!(
                "bad config: a voxel radius is at most {MAX_VOXEL_RADIUS}"
            ));
        }
        if config.entities > MAX_ENTITIES {
            return Err(format!("bad config: entities is at most {MAX_ENTITIES}"));
        }
        if !config.entity_radius.is_finite() || config.entity_radius < 0.0 {
            return Err("bad config: entity_radius must be finite and not negative".into());
        }
        Ok(config)
    }
}

/// The box of blocks an observation carries: from its lowest corner, x fastest, then z,
/// then y, the order a chunk section keeps its blocks in (a numpy array of shape
/// `(y, z, x)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VoxelBox {
    pub origin: [i32; 3],
    pub size: [u32; 3],
}
impl VoxelBox {
    /// The box `radius` blocks out from `centre` on each axis.
    pub fn around(centre: [i32; 3], radius: [u32; 3]) -> Self {
        let r = radius.map(|r| r.min(MAX_VOXEL_RADIUS));
        Self {
            origin: [0, 1, 2].map(|i| centre[i] - r[i] as i32),
            size: r.map(|r| 2 * r + 1),
        }
    }
    pub fn len(&self) -> usize {
        self.size.iter().map(|&s| s as usize).product()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Every position of the box, in its order.
    pub fn positions(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        let [x0, y0, z0] = self.origin;
        let [sx, sy, sz] = self.size.map(|s| s as i32);
        (0..sy).flat_map(move |y| {
            (0..sz).flat_map(move |z| (0..sx).map(move |x| [x0 + x, y0 + y, z0 + z]))
        })
    }
}

/// Block state ids as base64 of little-endian `u32`s (`numpy.frombuffer(..., "<u4")`).
pub fn encode_ids(ids: &[u32]) -> String {
    let bytes: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// A gym request's command word and the JSON after it (`gym-step {"forward": 1}`).
pub fn split_request(line: &str) -> (&str, &str) {
    let line = line.trim();
    match line.split_once(char::is_whitespace) {
        Some((word, rest)) => (word, rest.trim()),
        None => (line, ""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tick_marker_round_trips() {
        for tick in [0, 1, 255, 1 << 40, u64::MAX] {
            assert_eq!(read_tick_payload(&tick_payload(tick)), Some(tick));
        }
        assert_eq!(read_tick_payload(&[1, 2, 3]), None);
        assert_eq!(read_tick_payload(&[0; 9]), None);
    }

    /// The server starts exactly the ticks granted, in order, and the client sees each one
    /// finish; nothing runs before a grant.
    #[test]
    fn the_server_ticks_only_what_the_client_granted() {
        let gate = Arc::new(Lockstep::default());
        assert_eq!(gate.start(Duration::ZERO), None);
        let server = {
            let gate = gate.clone();
            std::thread::spawn(move || {
                let mut ticks = Vec::new();
                while ticks.len() < 5 {
                    if let Some(tick_ends) = gate.start(Duration::from_millis(50)) {
                        ticks.push((gate.finish(), tick_ends));
                    }
                }
                ticks
            })
        };
        let first = gate.grant(2, 2);
        assert_eq!(first, 2);
        assert!(gate.wait_finished(first, Duration::from_secs(10)));
        let second = gate.grant(3, 5);
        assert_eq!(second, 5);
        assert!(gate.wait_finished(second, Duration::from_secs(10)));
        let ticks = server.join().unwrap();
        assert_eq!(
            ticks.iter().map(|t| t.0).collect::<Vec<_>>(),
            [1, 2, 3, 4, 5]
        );
        // Each tick waits for at least the tick ends sent before its grant.
        assert!(ticks[..2].iter().all(|t| t.1 >= 2));
        assert!(ticks[2..].iter().all(|t| t.1 == 5));
        assert_eq!(gate.finished(), 5);
        assert_eq!(gate.start(Duration::ZERO), None);
    }

    #[test]
    fn waits_end_when_the_gate_closes_or_times_out() {
        let gate = Lockstep::default();
        let target = gate.grant(1, 0);
        assert!(!gate.wait_finished(target, Duration::from_millis(5)));
        gate.close();
        assert!(!gate.wait_finished(target, Duration::from_secs(10)));
        assert_eq!(gate.start(Duration::from_secs(10)), None);
    }

    #[test]
    fn a_dropped_server_closes_its_gate() {
        let gate = Arc::new(Lockstep::default());
        let target = gate.grant(1, 0);
        drop(CloseOnDrop(gate.clone()));
        assert!(!gate.wait_finished(target, Duration::from_secs(10)));
    }

    #[test]
    fn settling_lasts_until_the_next_grant() {
        let gate = Lockstep::default();
        assert!(!gate.settling());
        gate.settle();
        assert!(gate.settling());
        gate.grant(1, 0);
        assert!(!gate.settling());
    }

    /// A finish without a start (a server that lost its tick) never runs ahead of the starts.
    #[test]
    fn finishes_never_pass_starts() {
        let gate = Lockstep::default();
        assert_eq!(gate.finish(), 0);
        gate.grant(1, 0);
        assert!(gate.start(Duration::ZERO).is_some());
        assert_eq!(gate.finish(), 1);
        assert_eq!(gate.finish(), 1);
    }

    #[test]
    fn actions_take_flags_numbers_and_defaults() {
        let a = Action::parse(r#"{"forward": 1, "jump": true, "attack": 0.0, "use": 1, "yaw": 15.5, "pitch": -100, "hotbar": 3, "ticks": 4}"#).unwrap();
        assert!(a.forward && a.jump && a.use_item);
        assert!(!a.attack && !a.back && !a.sneak);
        assert_eq!(
            (a.yaw, a.pitch, a.hotbar, a.ticks),
            (15.5, -100.0, Some(3), 4)
        );
        assert_eq!(a.turn(170.0, 10.0), (185.5, -90.0));
        assert_eq!(Action::parse("").unwrap(), Action::default());
        assert_eq!(Action::parse("{}").unwrap().ticks, 1);
    }

    #[test]
    fn bad_actions_are_refused() {
        for bad in [
            r#"{"ticks": 0}"#,
            r#"{"ticks": 1000}"#,
            r#"{"hotbar": 0}"#,
            r#"{"hotbar": 10}"#,
            r#"{"fly": 1}"#,
            r#"{"forward": "yes"}"#,
            "not json",
        ] {
            assert!(Action::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn observation_configs_are_checked() {
        let config =
            ObsConfig::parse(&serde_json::json!({ "voxels": [2, 1, 2], "entities": 4 })).unwrap();
        assert_eq!(config.voxels, [2, 1, 2]);
        assert_eq!(config.entities, 4);
        assert!(config.inventory && config.chat);
        assert_eq!(
            ObsConfig::parse(&serde_json::json!({})).unwrap(),
            ObsConfig::default()
        );
        for bad in [
            serde_json::json!({ "voxels": [17, 0, 0] }),
            serde_json::json!({ "entities": 1000 }),
            serde_json::json!({ "entity_radius": -1.0 }),
            serde_json::json!({ "colour": true }),
        ] {
            assert!(ObsConfig::parse(&bad).is_err(), "{bad}");
        }
    }

    /// The box runs x fastest, then z, then y, from its lowest corner.
    #[test]
    fn voxel_boxes_follow_section_order() {
        let b = VoxelBox::around([10, 64, -3], [1, 0, 1]);
        assert_eq!(b.origin, [9, 64, -4]);
        assert_eq!(b.size, [3, 1, 3]);
        let p: Vec<_> = b.positions().collect();
        assert_eq!(p.len(), b.len());
        assert_eq!(p[0], [9, 64, -4]);
        assert_eq!(p[1], [10, 64, -4]);
        assert_eq!(p[3], [9, 64, -3]);
        assert_eq!(p[8], [11, 64, -2]);
        let tall = VoxelBox::around([0, 0, 0], [0, 1, 0]);
        assert_eq!(
            tall.positions().collect::<Vec<_>>(),
            [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        );
        assert!(VoxelBox::around([0, 0, 0], [99, 0, 0]).size[0] == 2 * MAX_VOXEL_RADIUS + 1);
    }

    #[test]
    fn ids_encode_as_little_endian_base64() {
        assert_eq!(encode_ids(&[]), "");
        // 1, 0x0102 → 01 00 00 00 02 01 00 00.
        assert_eq!(encode_ids(&[1, 0x0102]), "AQAAAAIBAAA=");
    }

    #[test]
    fn requests_split_into_word_and_body() {
        assert_eq!(
            split_request("gym-step {\"a\": 1}"),
            ("gym-step", "{\"a\": 1}")
        );
        assert_eq!(split_request("  gym-obs  "), ("gym-obs", ""));
        assert_eq!(split_request("gym-reset\t{}"), ("gym-reset", "{}"));
    }
}
