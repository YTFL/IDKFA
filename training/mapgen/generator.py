"""Procedural DOOM Map Generator across 4 Map Families.

Families:
- Family A (Combat Arena): 1-3 open rooms, varying enemies, ammo/health pickups.
- Family B (Multi-room Exploration): Graph of interconnected rooms, corridors, loops,
  dead ends, ordinary doors, and an exit.
- Family C (Keycard Puzzle): Keycard in a side branch, locked door of matching colour
  blocking the critical path to the exit.
- Family D (Full Gauntlet): Multi-room, combat ambushes, doors, keycard puzzles, and exit.

Includes automated BFS solvability checks (key reachable before locked door, exit reachable
after unlocking).
"""

from __future__ import annotations

import collections
import enum
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

from training.mapgen.udmf import (
    LOCK_BLUE,
    LOCK_RED,
    LOCK_YELLOW,
    SPECIAL_DOOR_LOCKED,
    SPECIAL_DOOR_RAISE,
    SPECIAL_EXIT_NORMAL,
    THING_BLUE_CARD,
    THING_CLIP,
    THING_DEMON,
    THING_IMP,
    THING_MEDIKIT,
    THING_PLAYER1,
    THING_RED_CARD,
    THING_SHELLS,
    THING_SHOTGUN,
    THING_SHOTGUN_GUY,
    THING_STIMPACK,
    THING_YELLOW_CARD,
    THING_ZOMBIEMAN,
    UDMFMap,
)

CELL_SIZE = 256.0
WALL_THICKNESS = 32.0

WALL_TEXTURES = ["STARTAN2", "STONE2", "BROWN1", "GRAY1", "BRICK12", "COMP2"]
FLOOR_TEXTURES = ["FLOOR0_1", "FLOOR4_8", "FLAT5_4", "GATE3", "FLOOR1_1"]
CEILING_TEXTURES = ["CEIL1_1", "TLITE6_4", "CEIL3_1", "FLAT1"]
DOOR_TEXTURES = ["BIGDOOR2", "DOORTRAK", "EXITDOOR"]


class MapFamily(str, enum.Enum):
    FAMILY_A = "combat_arena"
    FAMILY_B = "multi_room_exploration"
    FAMILY_C = "keycard_puzzle"
    FAMILY_D = "full_gauntlet"


@dataclass
class Room:
    id: int
    gx: int
    gy: int
    width: float = CELL_SIZE
    height: float = CELL_SIZE
    is_start: bool = False
    is_exit: bool = False
    key: int | None = None  # LOCK_RED, LOCK_BLUE, LOCK_YELLOW
    enemies: list[int] = field(default_factory=list)
    items: list[int] = field(default_factory=list)


@dataclass
class DoorEdge:
    r1: int
    r2: int
    locked: int = 0  # 0 = normal door, 1=RED, 2=BLUE, 3=YELLOW


class ProceduralMapGenerator:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate(
        self,
        family: MapFamily,
        grid_w: int = 4,
        grid_h: int = 4,
        num_rooms: int = 5,
        difficulty: float = 0.5,
    ) -> tuple[UDMFMap, bool]:
        """Generate a map of the specified family.
        
        Returns (UDMFMap, is_solvable).
        """
        if family == MapFamily.FAMILY_A:
            return self._generate_family_a(difficulty)
        elif family == MapFamily.FAMILY_B:
            return self._generate_family_b(grid_w, grid_h, num_rooms, difficulty)
        elif family == MapFamily.FAMILY_C:
            return self._generate_family_c(grid_w, grid_h, num_rooms, difficulty)
        elif family == MapFamily.FAMILY_D:
            return self._generate_family_d(grid_w, grid_h, num_rooms, difficulty)
        else:
            raise ValueError(f"Unknown family: {family}")

    # -------------------------------------------------------------------------
    # Family A: Combat Arena
    # -------------------------------------------------------------------------
    def _generate_family_a(self, difficulty: float) -> tuple[UDMFMap, bool]:
        # 1-3 rooms joined together
        m = UDMFMap()
        room_size = self.rng.choice([384.0, 512.0, 640.0])
        v0 = m.add_vertex(64.0, 64.0)
        v1 = m.add_vertex(64.0, 64.0 + room_size)
        v2 = m.add_vertex(64.0 + room_size, 64.0 + room_size)
        v3 = m.add_vertex(64.0 + room_size, 64.0)

        wall_tex = self.rng.choice(WALL_TEXTURES)
        floor_tex = self.rng.choice(FLOOR_TEXTURES)
        ceil_tex = self.rng.choice(CEILING_TEXTURES)

        sec = m.add_sector(
            heightfloor=0,
            heightceiling=144,
            texturefloor=floor_tex,
            textureceiling=ceil_tex,
            lightlevel=self.rng.choice([160, 192, 208]),
        )
        sd = m.add_sidedef(sector=sec, texturemiddle=wall_tex)

        m.add_linedef(v0, v1, sidefront=sd, blocking=True)
        m.add_linedef(v1, v2, sidefront=sd, blocking=True)
        m.add_linedef(v2, v3, sidefront=sd, blocking=True)
        m.add_linedef(v3, v0, sidefront=sd, blocking=True)

        # Player spawn
        m.add_thing(128.0, 128.0, THING_PLAYER1, angle=45)

        # Enemies scaled by difficulty
        enemy_count = int(1 + round(difficulty * 4))
        for _ in range(enemy_count):
            ex = self.rng.uniform(192.0, 64.0 + room_size - 64.0)
            ey = self.rng.uniform(192.0, 64.0 + room_size - 64.0)
            etype = self.rng.choices(
                [THING_ZOMBIEMAN, THING_SHOTGUN_GUY, THING_IMP, THING_DEMON],
                weights=[0.4, 0.2 + 0.2 * difficulty, 0.3 * difficulty, 0.2 * difficulty],
            )[0]
            m.add_thing(ex, ey, etype, angle=self.rng.choice([0, 90, 180, 270]))

        # Items
        m.add_thing(64.0 + room_size / 2, 64.0 + room_size / 2, THING_STIMPACK)
        if difficulty > 0.4:
            m.add_thing(64.0 + room_size / 2 + 32, 64.0 + room_size / 2, THING_CLIP)
            m.add_thing(128.0, 64.0 + room_size - 96.0, THING_MEDIKIT)

        return m, True

    # -------------------------------------------------------------------------
    # Grid Dungeon Layout Core (Families B, C, D)
    # -------------------------------------------------------------------------
    def _create_grid_dungeon(
        self,
        grid_w: int,
        grid_h: int,
        num_rooms: int,
        add_loops: bool = True,
    ) -> tuple[dict[int, Room], list[tuple[int, int]], list[tuple[int, int]]]:
        """Creates a connected graph of rooms on a 2D grid."""
        all_coords = [(x, y) for x in range(grid_w) for y in range(grid_h)]
        self.rng.shuffle(all_coords)
        selected_coords = all_coords[:num_rooms]

        rooms: dict[int, Room] = {}
        coord_to_id: dict[tuple[int, int], int] = {}
        for idx, (gx, gy) in enumerate(selected_coords):
            rooms[idx] = Room(id=idx, gx=gx, gy=gy)
            coord_to_id[(gx, gy)] = idx

        # Find potential adjacent neighbors
        adj: collections.defaultdict[int, list[int]] = collections.defaultdict(list)
        edges: list[tuple[int, int]] = []
        for r_id, r in rooms.items():
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor_coord = (r.gx + dx, r.gy + dy)
                if neighbor_coord in coord_to_id:
                    nbr_id = coord_to_id[neighbor_coord]
                    if nbr_id > r_id:
                        edges.append((r_id, nbr_id))

        if not edges and num_rooms > 1:
            # Force connect rooms by placing them in a chain if grid was too sparse
            return self._create_chain_dungeon(num_rooms)

        # Build spanning tree with random weights
        parent = list(range(num_rooms))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> bool:
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False

        self.rng.shuffle(edges)
        tree_edges: list[tuple[int, int]] = []
        cycle_edges: list[tuple[int, int]] = []

        for u, v in edges:
            if union(u, v):
                tree_edges.append((u, v))
            else:
                cycle_edges.append((u, v))

        # Add some loop edges if requested
        final_edges = list(tree_edges)
        if add_loops and cycle_edges:
            extra_loops = min(len(cycle_edges), self.rng.randint(1, 2))
            final_edges.extend(cycle_edges[:extra_loops])

        return rooms, tree_edges, final_edges

    def _create_chain_dungeon(
        self, num_rooms: int
    ) -> tuple[dict[int, Room], list[tuple[int, int]], list[tuple[int, int]]]:
        rooms = {}
        tree_edges = []
        for i in range(num_rooms):
            rooms[i] = Room(id=i, gx=i, gy=0)
            if i > 0:
                tree_edges.append((i - 1, i))
        return rooms, tree_edges, tree_edges

    # -------------------------------------------------------------------------
    # Family B: Multi-room Exploration
    # -------------------------------------------------------------------------
    def _generate_family_b(
        self, grid_w: int, grid_h: int, num_rooms: int, difficulty: float
    ) -> tuple[UDMFMap, bool]:
        rooms, _, edges = self._create_grid_dungeon(grid_w, grid_h, num_rooms, add_loops=True)

        # Room 0 is start, farthest room in tree BFS is exit
        start_id = 0
        rooms[start_id].is_start = True

        # BFS distances
        dist = {start_id: 0}
        q = collections.deque([start_id])
        adj = collections.defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        while q:
            curr = q.popleft()
            for nbr in adj[curr]:
                if nbr not in dist:
                    dist[nbr] = dist[curr] + 1
                    q.append(nbr)

        exit_id = max(dist.items(), key=lambda kv: kv[1])[0]
        rooms[exit_id].is_exit = True

        # Normal doors on some edges
        doors: list[DoorEdge] = []
        for u, v in edges:
            if self.rng.random() < 0.6:
                doors.append(DoorEdge(r1=u, r2=v, locked=0))

        # Place enemies and items
        for rid, room in rooms.items():
            if room.is_start:
                continue
            if self.rng.random() < (0.3 + 0.4 * difficulty):
                room.enemies.append(
                    self.rng.choice([THING_ZOMBIEMAN, THING_SHOTGUN_GUY, THING_IMP])
                )
            if self.rng.random() < 0.4:
                room.items.append(self.rng.choice([THING_STIMPACK, THING_CLIP]))

        solvable = self._verify_solvability(rooms, edges, doors, start_id, exit_id, key_needed=None)
        udmf = self._compile_dungeon(rooms, edges, doors)
        return udmf, solvable

    # -------------------------------------------------------------------------
    # Family C: Keycard Puzzle
    # -------------------------------------------------------------------------
    def _generate_family_c(
        self, grid_w: int, grid_h: int, num_rooms: int, difficulty: float
    ) -> tuple[UDMFMap, bool]:
        # Ensure at least 4 rooms for a proper branch + locked door
        num_rooms = max(num_rooms, 4)
        rooms, tree_edges, _ = self._create_grid_dungeon(grid_w, grid_h, num_rooms, add_loops=False)

        start_id = 0
        rooms[start_id].is_start = True

        # Build tree adjacency
        adj = collections.defaultdict(list)
        for u, v in tree_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Distances from start
        dist = {start_id: 0}
        pred = {start_id: None}
        q = collections.deque([start_id])
        while q:
            curr = q.popleft()
            for nbr in adj[curr]:
                if nbr not in dist:
                    dist[nbr] = dist[curr] + 1
                    pred[nbr] = curr
                    q.append(nbr)

        exit_id = max(dist.items(), key=lambda kv: kv[1])[0]
        rooms[exit_id].is_exit = True

        # Trace path from start to exit
        path = []
        curr = exit_id
        while curr is not None:
            path.append(curr)
            curr = pred[curr]
        path.reverse()

        # Place locked door on the path just before the exit (or 1 room before exit)
        locked_edge = None
        if len(path) >= 2:
            locked_edge = (path[-2], path[-1])

        # Pick key color
        key_color = self.rng.choice([LOCK_RED, LOCK_BLUE, LOCK_YELLOW])

        # Place key in a side room (a room NOT on path between locked door and exit)
        forbidden_set = {exit_id}
        available_rooms = [r for r in rooms.keys() if r not in forbidden_set and r != start_id]
        if not available_rooms:
            available_rooms = [start_id]
        key_room_id = self.rng.choice(available_rooms)
        rooms[key_room_id].key = key_color

        # Build door list
        doors: list[DoorEdge] = []
        if locked_edge:
            doors.append(DoorEdge(r1=locked_edge[0], r2=locked_edge[1], locked=key_color))

        # Add normal doors on other edges occasionally
        for u, v in tree_edges:
            if locked_edge and ((u, v) == locked_edge or (v, u) == locked_edge):
                continue
            if self.rng.random() < 0.4:
                doors.append(DoorEdge(r1=u, r2=v, locked=0))

        # Enemies and items
        for rid, room in rooms.items():
            if room.is_start:
                continue
            if self.rng.random() < (0.2 + 0.3 * difficulty):
                room.enemies.append(self.rng.choice([THING_ZOMBIEMAN, THING_IMP]))
            if self.rng.random() < 0.3:
                room.items.append(THING_STIMPACK)

        solvable = self._verify_solvability(
            rooms, tree_edges, doors, start_id, exit_id, key_needed=key_color
        )
        udmf = self._compile_dungeon(rooms, tree_edges, doors)
        return udmf, solvable

    # -------------------------------------------------------------------------
    # Family D: Full Gauntlet
    # -------------------------------------------------------------------------
    def _generate_family_d(
        self, grid_w: int, grid_h: int, num_rooms: int, difficulty: float
    ) -> tuple[UDMFMap, bool]:
        num_rooms = max(num_rooms, 6)
        rooms, tree_edges, edges = self._create_grid_dungeon(grid_w, grid_h, num_rooms, add_loops=True)

        start_id = 0
        rooms[start_id].is_start = True

        # BFS from start
        adj = collections.defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        dist = {start_id: 0}
        pred = {start_id: None}
        q = collections.deque([start_id])
        while q:
            curr = q.popleft()
            for nbr in adj[curr]:
                if nbr not in dist:
                    dist[nbr] = dist[curr] + 1
                    pred[nbr] = curr
                    q.append(nbr)

        exit_id = max(dist.items(), key=lambda kv: kv[1])[0]
        rooms[exit_id].is_exit = True

        key_color = self.rng.choice([LOCK_RED, LOCK_BLUE, LOCK_YELLOW])
        available = [r for r in rooms.keys() if r not in (start_id, exit_id)]
        key_room_id = self.rng.choice(available) if available else start_id
        rooms[key_room_id].key = key_color

        # Place locked door into the exit room
        exit_neighbors = adj[exit_id]
        doors: list[DoorEdge] = []
        gate_neighbor = None
        if exit_neighbors:
            gate_neighbor = exit_neighbors[0]
            doors.append(DoorEdge(r1=gate_neighbor, r2=exit_id, locked=key_color))

        # Add regular doors
        for u, v in edges:
            if gate_neighbor is not None and ((u, v) == (gate_neighbor, exit_id) or (v, u) == (gate_neighbor, exit_id)):
                continue
            if self.rng.random() < 0.5:
                doors.append(DoorEdge(r1=u, r2=v, locked=0))

        # Significant enemies and pickups
        for rid, room in rooms.items():
            if room.is_start:
                continue
            enemy_count = self.rng.randint(1, 2 if difficulty < 0.5 else 3)
            for _ in range(enemy_count):
                room.enemies.append(
                    self.rng.choices(
                        [THING_ZOMBIEMAN, THING_SHOTGUN_GUY, THING_IMP, THING_DEMON],
                        weights=[0.3, 0.3, 0.2 + 0.2 * difficulty, 0.2 * difficulty],
                    )[0]
                )
            if self.rng.random() < 0.6:
                room.items.append(
                    self.rng.choice([THING_STIMPACK, THING_MEDIKIT, THING_CLIP, THING_SHELLS])
                )

        # Place a shotgun in an intermediate room
        if available:
            rooms[available[0]].items.append(THING_SHOTGUN)

        solvable = self._verify_solvability(rooms, edges, doors, start_id, exit_id, key_needed=key_color)
        udmf = self._compile_dungeon(rooms, edges, doors)
        return udmf, solvable

    # -------------------------------------------------------------------------
    # Solvability Verification (BFS)
    # -------------------------------------------------------------------------
    def _verify_solvability(
        self,
        rooms: dict[int, Room],
        edges: list[tuple[int, int]],
        doors: list[DoorEdge],
        start_id: int,
        exit_id: int,
        key_needed: int | None,
    ) -> bool:
        """Graph BFS to verify path solvability.
        
        If key_needed is specified:
        1) Key room must be reachable without passing through the locked door.
        2) Exit must be reachable once the key is acquired.
        """
        door_map: dict[tuple[int, int], int] = {}
        for d in doors:
            door_map[(d.r1, d.r2)] = d.locked
            door_map[(d.r2, d.r1)] = d.locked

        adj = collections.defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)

        # Phase 1: Reachability without key
        def get_reachable(has_key: bool) -> set[int]:
            visited = {start_id}
            queue = collections.deque([start_id])
            while queue:
                curr = queue.popleft()
                for nbr in adj[curr]:
                    lock = door_map.get((curr, nbr), 0)
                    if lock > 0 and not has_key:
                        continue  # Blocked by locked door
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            return visited

        if key_needed is None:
            # Simple reachability to exit
            reachable = get_reachable(has_key=True)
            return exit_id in reachable

        # Keycard puzzle check:
        # Find room where key is located
        key_rooms = [r for r, room in rooms.items() if room.key == key_needed]
        if not key_rooms:
            return False
        target_key_room = key_rooms[0]

        reachable_before_key = get_reachable(has_key=False)
        if target_key_room not in reachable_before_key:
            return False  # Cannot reach key without opening the locked door!

        # After reaching key, can player reach exit?
        reachable_after_key = get_reachable(has_key=True)
        return exit_id in reachable_after_key

    # -------------------------------------------------------------------------
    # UDMF Compilation
    # -------------------------------------------------------------------------
    def _compile_dungeon(
        self,
        rooms: dict[int, Room],
        edges: list[tuple[int, int]],
        doors: list[DoorEdge],
    ) -> UDMFMap:
        m = UDMFMap()
        door_lookup: dict[tuple[int, int], int] = {}
        for d in doors:
            door_lookup[(d.r1, d.r2)] = d.locked
            door_lookup[(d.r2, d.r1)] = d.locked

        # Each room gets a sector
        room_sectors: dict[int, int] = {}
        room_coords: dict[int, tuple[float, float, float, float]] = {}

        base_x = 128.0
        base_y = 128.0
        margin = 64.0  # spacing between rooms for corridors/doors

        for rid, room in rooms.items():
            rx1 = base_x + room.gx * (CELL_SIZE + margin)
            ry1 = base_y + room.gy * (CELL_SIZE + margin)
            rx2 = rx1 + CELL_SIZE
            ry2 = ry1 + CELL_SIZE
            room_coords[rid] = (rx1, ry1, rx2, ry2)

            f_tex = self.rng.choice(FLOOR_TEXTURES)
            c_tex = self.rng.choice(CEILING_TEXTURES)
            sec = m.add_sector(
                heightfloor=0,
                heightceiling=128,
                texturefloor=f_tex,
                textureceiling=c_tex,
                lightlevel=192,
            )
            room_sectors[rid] = sec

        # Outer walls and connecting edges
        # Map out grid connections
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))

        for rid, room in rooms.items():
            rx1, ry1, rx2, ry2 = room_coords[rid]
            sec = room_sectors[rid]
            w_tex = self.rng.choice(WALL_TEXTURES)
            sd_room = m.add_sidedef(sector=sec, texturemiddle=w_tex)

            # Check 4 directions for connections
            for dx, dy, (x1, y1, x2, y2) in [
                (1, 0, (rx2, ry1, rx2, ry2)),    # East
                (-1, 0, (rx1, ry2, rx1, ry1)),   # West
                (0, 1, (rx2, ry2, rx1, ry2)),    # North
                (0, -1, (rx1, ry1, rx2, ry1)),   # South
            ]:
                nbr_id = None
                for other_id, other in rooms.items():
                    if other.gx == room.gx + dx and other.gy == room.gy + dy:
                        nbr_id = other_id
                        break

                has_connection = nbr_id is not None and (min(rid, nbr_id), max(rid, nbr_id)) in edge_set

                if not has_connection:
                    # Solid boundary wall
                    va = m.add_vertex(x1, y1)
                    vb = m.add_vertex(x2, y2)
                    m.add_linedef(va, vb, sidefront=sd_room, blocking=True)
                else:
                    # Connection exists: add doorway or open passage if this is smaller id
                    if rid < nbr_id:
                        self._build_corridor_doorway(
                            m,
                            room_coords[rid],
                            room_coords[nbr_id],
                            room_sectors[rid],
                            room_sectors[nbr_id],
                            door_lookup.get((rid, nbr_id)),
                        )

            # Populate Things in room
            cx = (rx1 + rx2) / 2
            cy = (ry1 + ry2) / 2

            if room.is_start:
                m.add_thing(cx, cy, THING_PLAYER1, angle=0)

            if room.is_exit:
                # Add exit trigger line across the center of exit room
                va = m.add_vertex(cx - 32, cy - 32)
                vb = m.add_vertex(cx + 32, cy + 32)
                sd_exit = m.add_sidedef(sector=sec, texturemiddle="-")
                m.add_linedef(
                    va, vb, sidefront=sd_exit, blocking=False, special=SPECIAL_EXIT_NORMAL, playercross=True
                )

            # Keycard
            if room.key:
                ktype = (
                    THING_RED_CARD
                    if room.key == LOCK_RED
                    else THING_BLUE_CARD
                    if room.key == LOCK_BLUE
                    else THING_YELLOW_CARD
                )
                m.add_thing(cx + 40, cy + 40, ktype)

            # Enemies
            for i, etype in enumerate(room.enemies):
                offset = (i + 1) * 35.0
                m.add_thing(cx - offset, cy + offset, etype, angle=180)

            # Items
            for j, itype in enumerate(room.items):
                offset = (j + 1) * 30.0
                m.add_thing(cx + offset, cy - offset, itype)

        return m

    def _build_corridor_doorway(
        self,
        m: UDMFMap,
        coords1: tuple[float, float, float, float],
        coords2: tuple[float, float, float, float],
        sec1: int,
        sec2: int,
        locked_type: int | None,
    ) -> None:
        """Builds a connecting passage or door between two adjacent rooms."""
        x1_a, y1_a, x2_a, y2_a = coords1
        x1_b, y1_b, x2_b, y2_b = coords2

        # Overlapping boundary
        door_w = 96.0
        if x2_a <= x1_b:  # Room 1 is West, Room 2 is East
            mid_y = (max(y1_a, y1_b) + min(y2_a, y2_b)) / 2
            v_a1 = m.add_vertex(x2_a, mid_y - door_w / 2)
            v_a2 = m.add_vertex(x2_a, mid_y + door_w / 2)
            v_b1 = m.add_vertex(x1_b, mid_y - door_w / 2)
            v_b2 = m.add_vertex(x1_b, mid_y + door_w / 2)
        elif y2_a <= y1_b:  # Room 1 is South, Room 2 is North
            mid_x = (max(x1_a, x1_b) + min(x2_a, x2_b)) / 2
            v_a1 = m.add_vertex(mid_x - door_w / 2, y2_a)
            v_a2 = m.add_vertex(mid_x + door_w / 2, y2_a)
            v_b1 = m.add_vertex(mid_x - door_w / 2, y1_b)
            v_b2 = m.add_vertex(mid_x + door_w / 2, y1_b)
        else:
            return

        if locked_type is None:
            # Open corridor
            sd_f = m.add_sidedef(sector=sec1, texturemiddle="-")
            sd_b = m.add_sidedef(sector=sec2, texturemiddle="-")
            m.add_linedef(v_a1, v_b2, sidefront=sd_f, sideback=sd_b, twosided=True, blocking=False)
        else:
            # Door sector (height 0 to 128)
            door_sec = m.add_sector(
                heightfloor=0,
                heightceiling=0,  # starts closed
                texturefloor="FLAT5_4",
                textureceiling="CEIL1_1",
                lightlevel=192,
            )
            sd_door_front = m.add_sidedef(sector=sec1, texturetop="BIGDOOR2", texturemiddle="-")
            sd_door_back = m.add_sidedef(sector=door_sec, texturetop="-", texturemiddle="-")

            special = SPECIAL_DOOR_LOCKED if locked_type > 0 else SPECIAL_DOOR_RAISE
            arg3 = locked_type if locked_type > 0 else 0

            m.add_linedef(
                v_a1,
                v_b2,
                sidefront=sd_door_front,
                sideback=sd_door_back,
                twosided=True,
                blocking=False,
                special=special,
                arg0=0,
                arg1=16,
                arg2=150,
                arg3=arg3,
                playeruse=True,
                monsteruse=True,
                repeatspecial=True,
                locknumber=locked_type,
            )
