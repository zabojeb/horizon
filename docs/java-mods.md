# Java mods and their loaders (Fabric, NeoForge, Forge): what can run, and how fast

Horizon has no Java classes, so a Fabric or NeoForge jar cannot load in it as it is ([mods.md](mods.md)). This
page weighs every way to bring Java mods to Horizon players anyway, what each one costs, how fast the result
would be, and what to build in which order. It complements [modpacks.md](modpacks.md) (porting packs) and
[native-hooks-neoforge.md](native-hooks-neoforge.md) (hooks named after NeoForge's events).

## Why a jar can't simply run

A Java mod is compiled against Minecraft's own classes (`net.minecraft.world.level.Level`,
`BlockBehaviour`...), and its loader changes those classes as they load:

- **Mixins.** Almost every Fabric mod, and many NeoForge ones, rewrite vanilla methods' bytecode
  (inject at HEAD, redirect a call, overwrite a method). The target is vanilla's bytecode; Horizon's code is
  Rust and has no such methods to inject into.
- **The API surface.** Mods call thousands of vanilla methods directly and subclass vanilla's classes
  (`Block`, `BlockEntity`, `Entity`, `Screen`). Their objects are compared by identity (`state.getBlock() ==
  Blocks.STONE`) and live in vanilla's object graph.
- **Loaders are patches on vanilla.** NeoForge's events are fired from patched vanilla methods; Fabric
  API's from mixins into them. Both need the vanilla classes to patch.

So "running a jar" means running vanilla's classes somewhere. The options differ in where.

## The options

### A. A JVM inside Horizon, with a Java facade over Horizon's world (not recommended)

Embed a JVM (JNI, or Java 22's foreign function interface), write a `net.minecraft.*` facade whose methods
call into Horizon's Rust state, and load mods against it.

- **Mixins don't apply**: they target method bodies the facade doesn't have. Every mixin would need a
  hand-written equivalent, mod by mod, which is the same work as a port.
- **The facade is a second Minecraft.** Mods reach deep into vanilla: block behaviour classes, entity AI
  goals, the renderer. Faking them faithfully is writing Minecraft again, in Java.
- **Slower than vanilla.** A world read that is a field load in vanilla becomes a native call: a JNI
  transition costs tens of nanoseconds, a foreign-function downcall a few; a modded tick makes millions of
  such reads (`getBlockState` per neighbour, per path node). Each Rust value crossing over needs a Java object
  with a stable identity (a `BlockState` interned per state, a `BlockPos` per call). Horizon's lead over
  vanilla (shorter ticks, a quarter of the memory, [server-next.md](server-next.md)) would go into the bridge.
- Legal: the facade cannot be written from vanilla's code, and NeoForge's and Forge's code may not be read
  (LGPL, [CONTRIBUTING](../CONTRIBUTING.md#only-original-code)).

### B. Horizon as the client of a modded Java server (recommended first)

Run the pack's server as it is (NeoForge or Fabric on Java), and let Horizon's client join it. The mods' logic
runs where it was written for; the client draws the world with Horizon's renderer.

What the client needs, all of it at the protocol level, none of it the mods' code:

1. **The loader's login handshake.** In configuration, NeoForge negotiates its payload channels and a Fabric
   server with Fabric API syncs its registries. A vanilla client is let in only when every channel the server
   requires is optional ([NeoForge: networking](https://docs.neoforged.net/docs/networking/)); a server with
   new blocks or items requires its registry sync. Horizon would answer both: list the channels it
   understands, and take the server's registry ids.
2. **Modded registries.** Vanilla's protocol sends block states and items as numbers from the client's own
   registry. A modded server numbers its blocks with the mods' included; the client must map the server's ids
   to its own. Content mods already append mod states after vanilla's ([modpacks.md](modpacks.md),
   "Registries 0.2"), so the map is a table per connection, built from the sync.
3. **The content itself.** Block and item declarations, models, textures, lang: what `horizon-port-pack
   carry` already extracts from the same jars ([modpacks.md](modpacks.md)), or the server's resource pack.
   A block the carry can't describe still gets the server's state id, drawn with the missing model.
4. **Mod payloads.** Custom packets of client-side features (a machine's GUI sync, a minimap's waypoints) are
   ignored, as a vanilla client ignores them: the feature is missing, the connection stays.

What works then: everything the server does (machines, recipes, worldgen, mobs, the mods' commands and
rules), vanilla's screens over modded containers (a chest-like grid), modded blocks and items with JSON
models. What doesn't: client code (custom renderers and animated models, mod screens, HUDs, keybinds, shaders).

**Speed.** The world ticks at the Java server's speed (with its performance mods: Lithium, ModernFix...);
the client renders at Horizon's ([benchmark.md](benchmark.md)), with Horizon's memory and load times.
The extra cost is one id lookup per block state and item on the wire. For a pack played on a server this is
most of Horizon's client-side win at a fraction of the cost of porting.

**Work.** A Fabric registry-sync answer (Fabric API is Apache-2.0: its source may be read), a NeoForge channel
negotiation from its public documentation and black-box captures (its source may not be read), an id map in
the client's chunk and item decoding, and a check in `horizon-port-pack` that the carried content matches the
server's registry. Each piece is a harness against a real modded server, as `test_dedicated_server_parity.py`
is against vanilla's.

### C. Ports: the mods rewritten as Horizon content and native mods (the long road, already under way)

[modpacks.md](modpacks.md): data and assets carry over mechanically (`.hmod`), logic is ported by an agent per
mod to a native mod on Horizon's hook points, each checked by a harness against the Java original. This is the
only option where both the server and the client run at Horizon's speed, single player included, and where
client-side features exist at all. Measured so far: All the Mods 10's 42 data-tier mods carry over and load;
logic, machine and deep tiers need the hook points and agents listed there.

**Speed.** Native: a ported mod costs what the same code costs in Horizon. With no mods loaded nothing
changes ("free when unused").

### D. Translating bytecode to Rust or WebAssembly (no)

Mechanical translation of a mod's classes still needs every vanilla class they call, so it is option A's
facade with a compiler in front, and Mixins still have nothing to apply to. Only pure-logic libraries (a math
or config library) would translate cleanly, and those are the mods that need no port at all.

### E. The Java instance next door (exists)

The launcher already creates and plays Minecraft: Java Edition instances with Fabric, Quilt, NeoForge or Forge
and installs their mods from Modrinth ([launcher/README.md](../launcher/README.md)). That is full compatibility
at vanilla's speed, and the fallback for anything A to D don't cover.

## Comparison

| | Runs the mods' own code | Client-side mod features | Server speed | Client speed | Cost |
|---|---|---|---|---|---|
| A. JVM + facade | partly (no Mixins) | partly | slower than vanilla | slower than vanilla | a second Minecraft |
| B. Horizon client, modded server | yes, on the server | no | Java (+ its perf mods) | Horizon | months: a handshake, an id map, harnesses |
| C. Ports | no (rewritten) | yes, when ported | Horizon | Horizon | per mod; agents + harnesses |
| D. Bytecode translation | no | no | n/a | n/a | not viable |
| E. Java instance | yes | yes | vanilla | vanilla | done |

## Recommendation

1. **B first, for servers.** It brings every server-side Java mod to Horizon's client at once, on both
   loaders, and its id map is the same table C needs for content mods in multiplayer ("Not in 0.2:
   multiplayer registry sync", [modpacks.md](modpacks.md)). Start with Fabric (readable source, simpler sync),
   then NeoForge from its documentation.
2. **C for single player and for the client side**, in the dependency order modpacks.md gives, with the hook
   points named after NeoForge's events so that a port reads like the original.
3. **Never A or D.** E stays the answer for what isn't covered yet.
