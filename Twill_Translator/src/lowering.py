from ir import WSSchedule
from lower_ir import (
    MPKProgram,
    BufferAlloc,
    WarpRegion,
    OpCall,
    Sync,
)


def lower_ws_to_mpk(schedule: WSSchedule) -> MPKProgram:
    program = MPKProgram()

    # 1. pipelines -> buffers
    for pipe in schedule.pipelines:
        program.buffers.append(
            BufferAlloc(
                name=pipe.buffer.name,
                stages=pipe.buffer.stages,
                scope=pipe.buffer.scope.name.lower(),
            )
        )

    # 2. group ops by warp
    warp_to_ops: dict[int, list[OpCall]] = {}

    for op in schedule.ops:
        warp_to_ops.setdefault(op.warp, []).append(
            OpCall(op=op.op)
        )

    for warp, ops in sorted(warp_to_ops.items()):
        program.warp_regions.append(
            WarpRegion(
                warp=warp,
                body=ops,
            )
        )

    # 3. WS sync -> lower sync
    for sync in schedule.syncs:
        program.syncs.append(
            Sync(
                pipeline=sync.pipeline,
                producer_warp=sync.producer_warp,
                consumer_warp=sync.consumer_warp,
            )
        )

    if program.buffers:
        program.k_stage = max(buf.stages for buf in program.buffers)

    return program