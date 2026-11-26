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
        m.add_thing(64.0 + room_size / 2 + 32, 64.0 + room_size / 2, THING_CLIP)
        if difficulty > 0.4:
            m.add_thing(128.0, 64.0 + room_size - 96.0, THING_MEDIKIT)

        m.metadata = {
            "family": MapFamily.FAMILY_A.value,
            "start_xy": (128.0, 128.0),
            "exit_xy": None,
            "key_xy": None,
            "key_color": None,
            "num_enemies": enemy_count,
        }
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
        """Creates a guaranteed-connected graph of rooms on a 2D grid via frontier growth."""
        while grid_w * grid_h < num_rooms:
            grid_w += 1
            grid_h += 1

        start_coord = (self.rng.randrange(grid_w), self.rng.randrange(grid_h))
        selected_coords: list[tuple[int, int]] = [start_coord]
        selected_set = {start_coord}
        frontier: list[tuple[int, int]] = []
        frontier_set: set[tuple[int, int]] = set()

        def _push_neighbors(cx: int, cy: int) -> None:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < grid_w and 0 <= ny < grid_h:
                    nc = (nx, ny)
                    if nc not in selected_set and nc not in frontier_set:
                        frontier.append(nc)
                        frontier_set.add(nc)

        _push_neighbors(*start_coord)
        while len(selected_coords) < num_rooms and frontier:
            pick = self.rng.randrange(len(frontier))
            coord = frontier.pop(pick)
            frontier_set.remove(coord)
            selected_coords.append(coord)
            selected_set.add(coord)
            _push_neighbors(*coord)

        rooms: dict[int, Room] = {}
        coord_to_id: dict[tuple[int, int], int] = {}
        for idx, (gx, gy) in enumerate(selected_coords):
            rooms[idx] = Room(id=idx, gx=gx, gy=gy)
            coord_to_id[(gx, gy)] = idx

        # Find potential adjacent neighbors
        edges: list[tuple[int, int]] = []
        for r_id, r in rooms.items():
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor_coord = (r.gx + dx, r.gy + dy)
                if neighbor_coord in coord_to_id:
                    nbr_id = coord_to_id[neighbor_coord]
                    if nbr_id > r_id:
                        edges.append((r_id, nbr_id))

        if not edges and len(rooms) > 1:
            return self._create_chain_dungeon(len(rooms))

        # Build spanning tree with random weights
        parent = list(range(len(rooms)))

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

        # BFS on spanning tree so exit_id is guaranteed to be a tree leaf (degree 1)
        tree_adj = collections.defaultdict(list)
        for u, v in tree_edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)

        dist = {start_id: 0}
        q = collections.deque([start_id])
        while q:
            curr = q.popleft()
            for nbr in tree_adj[curr]:
                if nbr not in dist:
                    dist[nbr] = dist[curr] + 1
                    q.append(nbr)

        exit_id = max(dist.items(), key=lambda kv: kv[1])[0]
        rooms[exit_id].is_exit = True

        # Keep loops elsewhere, but ensure exit_id has degree 1 (via its tree edge)
        # so the single locked door strictly gates the exit room.
        tree_set = {(min(u, v), max(u, v)) for u, v in tree_edges}
        edges = [
            (u, v)
            for u, v in edges
            if (exit_id not in (u, v)) or ((min(u, v), max(u, v)) in tree_set)
        ]

        # Recompute exit_id's sole remaining neighbor
        gate_neighbor = None
        for u, v in edges:
            if u == exit_id:
                gate_neighbor = v
                break
            if v == exit_id:
                gate_neighbor = u
                break

        key_color = self.rng.choice([LOCK_RED, LOCK_BLUE, LOCK_YELLOW])
        available = [r for r in rooms.keys() if r not in (start_id, exit_id)]
        key_room_id = self.rng.choice(available) if available else start_id
        rooms[key_room_id].key = key_color

        # Place locked door into the exit room
        doors: list[DoorEdge] = []
        if gate_neighbor is not None:
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
        if exit_id in reachable_before_key:
            return False  # Exit must be gated behind the locked door!

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

        m.metadata["room_centers"] = {
            int(rid): ((rx1 + rx2) / 2.0, (ry1 + ry2) / 2.0)
            for rid, (rx1, ry1, rx2, ry2) in room_coords.items()
        }
        m.metadata["room_bounds"] = {
            int(rid): (float(rx1), float(ry1), float(rx2), float(ry2))
            for rid, (rx1, ry1, rx2, ry2) in room_coords.items()
        }
        m.metadata["nav_edges"] = [
            (int(u), int(v), int(door_lookup[(u, v)]) if (u, v) in door_lookup else -1)
            for u, v in edges
        ]

        # Outer walls and connecting edges
        # Map out grid connections
        edge_set = set()
        for u, v in edges:
            edge_set.add((min(u, v), max(u, v)))

        door_w = 96.0
        half_w = door_w / 2.0

        # Build lookup from (gx, gy) to room id for O(1) neighbor checks
        grid_to_id = {(r.gx, r.gy): rid for rid, r in rooms.items()}
        # Store shared portal vertices per undirected room edge so room walls and corridors share exact vertices
        portal_verts: dict[tuple[int, int], tuple[int, int, int, int]] = {}

        def _get_portal_verts(r_a: int, r_b: int) -> tuple[int, int, int, int]:
            key = (min(r_a, r_b), max(r_a, r_b))
            if key in portal_verts:
                return portal_verts[key]
            # Normalize so ra is West or South of rb
            ra_id, rb_id = key
            if rooms[ra_id].gx > rooms[rb_id].gx or rooms[ra_id].gy > rooms[rb_id].gy:
                ra_id, rb_id = rb_id, ra_id
            x1_a, y1_a, x2_a, y2_a = room_coords[ra_id]
            x1_b, y1_b, x2_b, y2_b = room_coords[rb_id]
            if rooms[ra_id].gx < rooms[rb_id].gx:
                mid_y = (max(y1_a, y1_b) + min(y2_a, y2_b)) / 2.0
                va_low = m.add_vertex(x2_a, mid_y - half_w)
                va_high = m.add_vertex(x2_a, mid_y + half_w)
                vb_low = m.add_vertex(x1_b, mid_y - half_w)
                vb_high = m.add_vertex(x1_b, mid_y + half_w)
            else:
                mid_x = (max(x1_a, x1_b) + min(x2_a, x2_b)) / 2.0
                va_low = m.add_vertex(mid_x - half_w, y2_a)
                va_high = m.add_vertex(mid_x + half_w, y2_a)
                vb_low = m.add_vertex(mid_x - half_w, y1_b)
                vb_high = m.add_vertex(mid_x + half_w, y1_b)
            portal_verts[key] = (va_low, va_high, vb_low, vb_high)
            return portal_verts[key]

        for rid, room in rooms.items():
            rx1, ry1, rx2, ry2 = room_coords[rid]
            sec = room_sectors[rid]
            w_tex = self.rng.choice(WALL_TEXTURES)
            sd_room = m.add_sidedef(sector=sec, texturemiddle=w_tex)

            # Room corner vertices in clockwise order so sidefront faces INSIDE the room
            v_bl = m.add_vertex(rx1, ry1)
            v_tl = m.add_vertex(rx1, ry2)
            v_tr = m.add_vertex(rx2, ry2)
            v_br = m.add_vertex(rx2, ry1)

            # 4 clockwise edges: West (+y), North (+x), East (-y), South (-x)
            for dx, dy, v_start, v_end, is_low_side in [
                (-1, 0, v_bl, v_tl, False),  # West: room is East of neighbor
                (0, 1, v_tl, v_tr, True),    # North: room is South of neighbor
                (1, 0, v_tr, v_br, True),    # East: room is West of neighbor
                (0, -1, v_br, v_bl, False),  # South: room is North of neighbor
            ]:
                nbr_id = grid_to_id.get((room.gx + dx, room.gy + dy))
                has_connection = nbr_id is not None and (min(rid, nbr_id), max(rid, nbr_id)) in edge_set

                if not has_connection:
                    m.add_linedef(v_start, v_end, sidefront=sd_room, blocking=True)
                else:
                    va_low, va_high, vb_low, vb_high = _get_portal_verts(rid, nbr_id)
                    p_low, p_high = (va_low, va_high) if is_low_side else (vb_low, vb_high)
                    # Maintain clockwise room perimeter around the doorway opening
                    if (dx, dy) in ((-1, 0), (0, 1)):
                        # Increasing coordinate along edge: v_start -> p_low, p_high -> v_end
                        m.add_linedef(v_start, p_low, sidefront=sd_room, blocking=True)
                        m.add_linedef(p_high, v_end, sidefront=sd_room, blocking=True)
                    else:
                        # Decreasing coordinate along edge: v_start -> p_high, p_low -> v_end
                        m.add_linedef(v_start, p_high, sidefront=sd_room, blocking=True)
                        m.add_linedef(p_low, v_end, sidefront=sd_room, blocking=True)

                    if rid < nbr_id:
                        self._build_corridor_doorway(
                            m,
                            room,
                            rooms[nbr_id],
                            room_sectors[rid],
                            room_sectors[nbr_id],
                            _get_portal_verts(rid, nbr_id),
                            door_lookup.get((rid, nbr_id)),
                            w_tex,
                        )

            # Populate Things in room
            cx = (rx1 + rx2) / 2
            cy = (ry1 + ry2) / 2

            if room.is_start:
                m.add_thing(cx, cy, THING_PLAYER1, angle=0)
                m.metadata["start_xy"] = (float(cx), float(cy))

            if room.is_exit:
                # Two-sided walkover exit cross at the center of the exit room
                sd_exit = m.add_sidedef(sector=sec, texturemiddle="-")
                ve1 = m.add_vertex(cx - 48, cy)
                ve2 = m.add_vertex(cx + 48, cy)
                ve3 = m.add_vertex(cx, cy - 48)
                ve4 = m.add_vertex(cx, cy + 48)
                m.add_linedef(
                    ve1, ve2, sidefront=sd_exit, sideback=sd_exit, twosided=True,
                    blocking=False, special=SPECIAL_EXIT_NORMAL, playercross=True,
                )
                m.add_linedef(
                    ve3, ve4, sidefront=sd_exit, sideback=sd_exit, twosided=True,
                    blocking=False, special=SPECIAL_EXIT_NORMAL, playercross=True,
                )
                m.metadata["exit_xy"] = (float(cx), float(cy))

            # Keycard
            if room.key:
                ktype = (
                    THING_RED_CARD
                    if room.key == LOCK_RED
                    else THING_BLUE_CARD
                    if room.key == LOCK_BLUE
                    else THING_YELLOW_CARD
                )
                kx, ky = float(cx + 40), float(cy + 40)
                m.add_thing(kx, ky, ktype)
                m.metadata["key_xy"] = (kx, ky)
                m.metadata["key_color"] = int(room.key)

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
        room1: Room,
        room2: Room,
        sec1: int,
        sec2: int,
        p_verts: tuple[int, int, int, int],
        locked_type: int | None,
        wall_tex: str,
    ) -> None:
        """Builds a watertight corridor or door sector between two adjacent rooms."""
        # Normalize so r_low is West (smaller gx) or South (smaller gy) of r_high
        if room1.gx > room2.gx or room1.gy > room2.gy:
            sec_low, sec_high = sec2, sec1
            is_east_west = room2.gx < room1.gx
        else:
            sec_low, sec_high = sec1, sec2
            is_east_west = room1.gx < room2.gx

        va_low, va_high, vb_low, vb_high = p_verts

        if locked_type is None:
            mid_sec = m.add_sector(
                heightfloor=0,
                heightceiling=128,
                texturefloor="FLOOR0_1",
                textureceiling="CEIL1_1",
                lightlevel=176,
            )
            side_wall_tex = wall_tex
        else:
            mid_sec = m.add_sector(
                heightfloor=0,
                heightceiling=0,  # starts closed
                texturefloor="FLAT5_4",
                textureceiling="CEIL1_1",
                lightlevel=192,
            )
            side_wall_tex = "DOORTRAK"

        sd_mid_wall = m.add_sidedef(sector=mid_sec, texturemiddle=side_wall_tex)

        # Solid side walls of the corridor/door sector (sidefront faces INSIDE mid_sec)
        if is_east_west:
            # East-West: va_high -> vb_high (+x, right=-y=inside), vb_low -> va_low (-x, right=+y=inside)
            m.add_linedef(va_high, vb_high, sidefront=sd_mid_wall, blocking=True)
            m.add_linedef(vb_low, va_low, sidefront=sd_mid_wall, blocking=True)
            # Portals with sidefront facing OUTWARD into sec_low / sec_high (required for playeruse)
            p1_v1, p1_v2 = va_high, va_low  # -y -> right=-x=sec_low
            p2_v1, p2_v2 = vb_low, vb_high  # +y -> right=+x=sec_high
        else:
            # South-North: va_low -> vb_low (+y, right=+x=inside), vb_high -> va_high (-y, right=-x=inside)
            m.add_linedef(va_low, vb_low, sidefront=sd_mid_wall, blocking=True)
            m.add_linedef(vb_high, va_high, sidefront=sd_mid_wall, blocking=True)
            # Portals with sidefront facing OUTWARD into sec_low / sec_high
            p1_v1, p1_v2 = va_low, va_high  # +x -> right=-y=sec_low
            p2_v1, p2_v2 = vb_high, vb_low  # -x -> right=+y=sec_high

        if locked_type is None:
            sd_low = m.add_sidedef(sector=sec_low, texturemiddle="-")
            sd_high = m.add_sidedef(sector=sec_high, texturemiddle="-")
            sd_mid_p = m.add_sidedef(sector=mid_sec, texturemiddle="-")
            m.add_linedef(p1_v1, p1_v2, sidefront=sd_low, sideback=sd_mid_p, twosided=True, blocking=False)
            m.add_linedef(p2_v1, p2_v2, sidefront=sd_high, sideback=sd_mid_p, twosided=True, blocking=False)
        else:
            special = SPECIAL_DOOR_LOCKED if locked_type > 0 else SPECIAL_DOOR_RAISE
            arg3 = locked_type if locked_type > 0 else 0
            sd_low = m.add_sidedef(sector=sec_low, texturetop="BIGDOOR2", texturemiddle="-")
            sd_high = m.add_sidedef(sector=sec_high, texturetop="BIGDOOR2", texturemiddle="-")
            sd_mid_d = m.add_sidedef(sector=mid_sec, texturetop="-", texturemiddle="-")

            for pv1, pv2, sd_outer in ((p1_v1, p1_v2, sd_low), (p2_v1, p2_v2, sd_high)):
                m.add_linedef(
                    pv1,
                    pv2,
                    sidefront=sd_outer,
                    sideback=sd_mid_d,
                    twosided=True,
                    blocking=False,
                    special=special,
                    arg0=0,
                    arg1=16,
                    arg2=150,
                    arg3=arg3,
                    playeruse=True,
                    monsteruse=(locked_type == 0),
                    repeatspecial=True,
                    locknumber=locked_type,
                )
