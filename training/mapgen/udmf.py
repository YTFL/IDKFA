"""UDMF (Universal Doom Map Format) map writer and compiler.

Generates ZDoom-compatible UDMF textmaps and writes PWAD files directly
via doom_env.wad without external tool dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from doom_env.wad import Lump, write_wad


@dataclass
class Vertex:
    x: float
    y: float


@dataclass
class Sector:
    texturefloor: str = "FLOOR0_1"
    textureceiling: str = "CEIL1_1"
    heightfloor: int = 0
    heightceiling: int = 128
    lightlevel: int = 192
    tag: int = 0


@dataclass
class Sidedef:
    sector: int
    texturetop: str = "-"
    texturemiddle: str = "-"
    texturebottom: str = "-"


@dataclass
class Linedef:
    v1: int
    v2: int
    sidefront: int
    sideback: int = -1
    twosided: bool = False
    blocking: bool = True
    dontpegbottom: bool = True
    special: int = 0
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0
    arg3: int = 0
    arg4: int = 0
    playeruse: bool = False
    monsteruse: bool = False
    playercross: bool = False
    repeatspecial: bool = False
    locknumber: int = 0


@dataclass
class Thing:
    x: float
    y: float
    angle: int = 0
    type: int = 1
    skill1: bool = True
    skill2: bool = True
    skill3: bool = True
    skill4: bool = True
    skill5: bool = True
    single: bool = True


# Standard Doom Thing Types
THING_PLAYER1 = 1
THING_ZOMBIEMAN = 3004
THING_SHOTGUN_GUY = 9
THING_IMP = 3001
THING_DEMON = 3002

# Pickups
THING_STIMPACK = 2011
THING_MEDIKIT = 2012
THING_CLIP = 2007
THING_SHELLS = 2008
THING_SHOTGUN = 2001

# Keycards & Skulls
THING_BLUE_CARD = 5
THING_YELLOW_CARD = 6
THING_RED_CARD = 13
THING_BLUE_SKULL = 40
THING_YELLOW_SKULL = 39
THING_RED_SKULL = 38

# UDMF / ZDoom Specials
SPECIAL_DOOR_RAISE = 12         # Door_Raise(tag, speed, delay)
SPECIAL_DOOR_LOCKED = 13        # Door_LockedRaise(tag, speed, delay, lock)
SPECIAL_EXIT_NORMAL = 243       # Exit_Normal(position)

LOCK_RED = 1
LOCK_BLUE = 2
LOCK_YELLOW = 3


class UDMFMap:
    def __init__(self, name: str = "MAP01"):
        self.name = name
        self.vertices: list[Vertex] = []
        self.sectors: list[Sector] = []
        self.sidedefs: list[Sidedef] = []
        self.linedefs: list[Linedef] = []
        self.things: list[Thing] = []

    def add_vertex(self, x: float, y: float) -> int:
        idx = len(self.vertices)
        self.vertices.append(Vertex(x, y))
        return idx

    def add_sector(
        self,
        heightfloor: int = 0,
        heightceiling: int = 128,
        texturefloor: str = "FLOOR0_1",
        textureceiling: str = "CEIL1_1",
        lightlevel: int = 192,
        tag: int = 0,
    ) -> int:
        idx = len(self.sectors)
        self.sectors.append(
            Sector(
                texturefloor=texturefloor,
                textureceiling=textureceiling,
                heightfloor=heightfloor,
                heightceiling=heightceiling,
                lightlevel=lightlevel,
                tag=tag,
            )
        )
        return idx

    def add_sidedef(
        self,
        sector: int,
        texturemiddle: str = "-",
        texturetop: str = "-",
        texturebottom: str = "-",
    ) -> int:
        idx = len(self.sidedefs)
        self.sidedefs.append(
            Sidedef(
                sector=sector,
                texturetop=texturetop,
                texturemiddle=texturemiddle,
                texturebottom=texturebottom,
            )
        )
        return idx

    def add_linedef(
        self,
        v1: int,
        v2: int,
        sidefront: int,
        sideback: int = -1,
        twosided: bool = False,
        blocking: bool = True,
        special: int = 0,
        arg0: int = 0,
        arg1: int = 0,
        arg2: int = 0,
        arg3: int = 0,
        arg4: int = 0,
        playeruse: bool = False,
        monsteruse: bool = False,
        playercross: bool = False,
        repeatspecial: bool = False,
        locknumber: int = 0,
    ) -> int:
        idx = len(self.linedefs)
        self.linedefs.append(
            Linedef(
                v1=v1,
                v2=v2,
                sidefront=sidefront,
                sideback=sideback,
                twosided=twosided,
                blocking=blocking,
                special=special,
                arg0=arg0,
                arg1=arg1,
                arg2=arg2,
                arg3=arg3,
                arg4=arg4,
                playeruse=playeruse,
                monsteruse=monsteruse,
                playercross=playercross,
                repeatspecial=repeatspecial,
                locknumber=locknumber,
            )
        )
        return idx

    def add_thing(self, x: float, y: float, thing_type: int, angle: int = 0) -> int:
        idx = len(self.things)
        self.things.append(Thing(x=x, y=y, angle=angle, type=thing_type))
        return idx

    def to_textmap(self) -> str:
        lines: list[str] = ['namespace = "zdoom";\n']
        for v in self.vertices:
            lines.append(f"vertex {{ x = {v.x:.1f}; y = {v.y:.1f}; }}\n")
        for s in self.sectors:
            sec_str = (
                f'sector {{ texturefloor = "{s.texturefloor}"; textureceiling = "{s.textureceiling}"; '
                f"heightfloor = {s.heightfloor}; heightceiling = {s.heightceiling}; "
                f"lightlevel = {s.lightlevel};"
            )
            if s.tag:
                sec_str += f" id = {s.tag};"
            sec_str += " }\n"
            lines.append(sec_str)
        for side in self.sidedefs:
            lines.append(
                f'sidedef {{ sector = {side.sector}; texturetop = "{side.texturetop}"; '
                f'texturemiddle = "{side.texturemiddle}"; texturebottom = "{side.texturebottom}"; }}\n'
            )
        for ld in self.linedefs:
            parts = [f"linedef {{ v1 = {ld.v1}; v2 = {ld.v2}; sidefront = {ld.sidefront};"]
            if ld.sideback >= 0:
                parts.append(f"sideback = {ld.sideback};")
            if ld.twosided:
                parts.append("twosided = true;")
            if ld.blocking:
                parts.append("blocking = true;")
            if ld.dontpegbottom:
                parts.append("dontpegbottom = true;")
            if ld.special:
                parts.append(f"special = {ld.special};")
                if ld.arg0:
                    parts.append(f"arg0 = {ld.arg0};")
                if ld.arg1:
                    parts.append(f"arg1 = {ld.arg1};")
                if ld.arg2:
                    parts.append(f"arg2 = {ld.arg2};")
                if ld.arg3:
                    parts.append(f"arg3 = {ld.arg3};")
                if ld.arg4:
                    parts.append(f"arg4 = {ld.arg4};")
            if ld.playeruse:
                parts.append("playeruse = true;")
            if ld.monsteruse:
                parts.append("monsteruse = true;")
            if ld.playercross:
                parts.append("playercross = true;")
            if ld.repeatspecial:
                parts.append("repeatspecial = true;")
            if ld.locknumber:
                parts.append(f"locknumber = {ld.locknumber};")
            parts.append("}\n")
            lines.append(" ".join(parts))
        for th in self.things:
            lines.append(
                f"thing {{ x = {th.x:.1f}; y = {th.y:.1f}; angle = {th.angle}; type = {th.type}; "
                f"skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }}\n"
            )
        return "".join(lines)

    def write_wad(self, path: str | Path) -> None:
        text = self.to_textmap().encode("ascii")
        lumps = [
            Lump(self.name, b""),
            Lump("TEXTMAP", text),
            Lump("ENDMAP", b""),
        ]
        write_wad(path, lumps, kind=b"PWAD")
