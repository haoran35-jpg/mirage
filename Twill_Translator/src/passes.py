from ir import WSSchedule, WSSync


def get_warp(schedule: WSSchedule, op_name: str) -> int:
    for op in schedule.ops:
        if op.op == op_name:
            return op.warp

    raise ValueError(f"Unknown op: {op_name}")


def verify_schedule(schedule: WSSchedule) -> None:
    op_names = {op.op for op in schedule.ops}

    if len(op_names) != len(schedule.ops):
        raise ValueError("Duplicate op names")

    for op in schedule.ops:
        if op.warp < 0:
            raise ValueError(
                f"Invalid warp id for {op.op}: {op.warp}"
            )

    for pipe in schedule.pipelines:
        if pipe.producer not in op_names:
            raise ValueError(
                f"Unknown producer: {pipe.producer}"
            )

        if pipe.consumer not in op_names:
            raise ValueError(
                f"Unknown consumer: {pipe.consumer}"
            )

        if pipe.buffer.stages <= 0:
            raise ValueError(
                f"Invalid pipeline depth for "
                f"{pipe.buffer.name}"
            )

def insert_sync(schedule: WSSchedule) -> WSSchedule:
    syncs = []

    for pipe in schedule.pipelines:
        producer_warp = get_warp(
            schedule,
            pipe.producer
        )

        consumer_warp = get_warp(
            schedule,
            pipe.consumer
        )

        # Same warp executes in program order,
        # so no cross-warp synchronization is needed.
        if producer_warp == consumer_warp:
            continue

        syncs.append(
            WSSync(
                pipeline=pipe.name,
                producer_warp=producer_warp,
                consumer_warp=consumer_warp,
            )
        )

    schedule.syncs = syncs
    return schedule