from ir import (
    MemoryScope,
    WSOp,
    WSBuffer,
    WSPipeline,
    WSSchedule,
)


def import_opw(opw: dict[str, int]) -> list[WSOp]:
    """
    Lower Twill op-to-warp assignments into WSOp nodes.

    Example:
        opw["tma_K"] = 0

    becomes:
        WSOp(op="tma_K", warp=0)
    """

    result: list[WSOp] = []

    for op, warp in opw.items():
        result.append(
            WSOp(
                op=op,
                warp=warp,
            )
        )

    return result


def infer_depth(
    op_liveness: dict[int, list[int]],
) -> int:
    """
    Infer minimum pipeline depth from tile liveness.

    Input semantic:

        op_liveness[i] = list of time slots during
                         which iteration i's output tile is live.

    Example:

        {
            0: [0, 1, 2],
            1: [1, 2, 3],
            2: [2, 3, 4],
        }

    At t=2, tiles from iterations 0, 1, and 2
    are all simultaneously live, therefore depth = 3.
    """

    live_count: dict[int, int] = {}

    for iteration, times in op_liveness.items():
        for t in times:
            live_count[t] = live_count.get(t, 0) + 1

    if not live_count:
        return 1

    return max(live_count.values())


def import_liveness(
    liveness: dict[str, dict[int, list[int]]],
    pipeline_info: dict[str, tuple[str, str, str]],
) -> list[WSPipeline]:
    """
    Lower Twill liveness results into WSPipeline nodes.

    pipeline_info maps:

        producer op
            ->
        (pipeline name, buffer name, consumer op)

    Example:

        "tma_K":
            ("K_pipeline", "K_smem", "mma_QK")
    """

    pipelines: list[WSPipeline] = []

    for producer, op_liveness in liveness.items():

        if producer not in pipeline_info:
            raise ValueError(
                f"Missing pipeline metadata for producer '{producer}'"
            )

        depth = infer_depth(op_liveness)

        pipeline_name, buffer_name, consumer = pipeline_info[producer]

        buffer = WSBuffer(
            name=buffer_name,
            stages=depth,
            scope=MemoryScope.SHARED,
        )

        pipeline = WSPipeline(
            name=pipeline_name,
            producer=producer,
            consumer=consumer,
            buffer=buffer,
        )

        pipelines.append(pipeline)

    return pipelines


def import_twill(
    opw: dict[str, int],
    liveness: dict[str, dict[int, list[int]]],
    pipeline_info: dict[str, tuple[str, str, str]],
) -> WSSchedule:
    """
    Main frontend entry point.

    Twill output
        ->
    canonical WS schedule IR.
    """

    ops = import_opw(opw)

    pipelines = import_liveness(
        liveness,
        pipeline_info,
    )

    return WSSchedule(
        ops=ops,
        pipelines=pipelines,
    )