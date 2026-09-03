from dataclasses import dataclass, field
from enum import Enum, auto


class MemoryScope(Enum):
    SHARED = auto()
    TMEM = auto()
    REGISTER = auto()


@dataclass(frozen=True)
class WSOp:
    op: str
    warp: int


@dataclass(frozen=True)
class WSBuffer:
    name: str
    stages: int
    scope: MemoryScope = MemoryScope.SHARED


@dataclass(frozen=True)
class WSPipeline:
    name: str
    producer: str
    consumer: str
    buffer: WSBuffer


@dataclass(frozen=True)
class WSSync:
    pipeline: str
    producer_warp: int
    consumer_warp: int


@dataclass
class WSSchedule:
    ops: list[WSOp] = field(default_factory=list)
    pipelines: list[WSPipeline] = field(default_factory=list)
    syncs: list[WSSync] = field(default_factory=list)