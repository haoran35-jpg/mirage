from dataclasses import dataclass, field


@dataclass
class Stmt:
    pass


@dataclass
class WarpRegion(Stmt):
    warp: int
    body: list["Stmt"] = field(default_factory=list)


@dataclass
class BufferAlloc(Stmt):
    name: str
    stages: int
    scope: str


@dataclass
class OpCall(Stmt):
    op: str


@dataclass
class Sync(Stmt):
    pipeline: str
    producer_warp: int
    consumer_warp: int


@dataclass
class MPKProgram:
    buffers: list[BufferAlloc] = field(default_factory=list)
    warp_regions: list[WarpRegion] = field(default_factory=list)
    syncs: list[Sync] = field(default_factory=list)