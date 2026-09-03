from twill_importer import import_twill
from passes import verify_schedule, insert_sync
from lowering import lower_ws_to_mpk
from codegen import emit_cuda

opw = {
    "tma_K": 0,
    "tma_V": 0,
    "mma_QK": 1,
    "soft": 1,
    "mma_PV": 2,
}


liveness = {
    "tma_K": {
        0: [0, 1, 2],
        1: [1, 2, 3],
        2: [2, 3, 4],
    },

    "tma_V": {
        0: [0, 1],
        1: [1, 2],
        2: [2, 3],
    },
}


pipeline_info = {
    "tma_K": (
        "K_pipeline",
        "K_smem",
        "mma_QK",
    ),

    "tma_V": (
        "V_pipeline",
        "V_smem",
        "mma_PV",
    ),
}


# 1. Twill -> WS Schedule IR
schedule = import_twill(
    opw=opw,
    liveness=liveness,
    pipeline_info=pipeline_info,
)

# 2. Verify
verify_schedule(schedule)

# 3. Insert synchronization
schedule = insert_sync(schedule)

# 4. WS Schedule IR -> MPK Lower IR
program = lower_ws_to_mpk(schedule)

cuda = emit_cuda(program)



print("=== Ops ===")
for op in schedule.ops:
    print(op)


print("\n=== Pipelines ===")
for pipe in schedule.pipelines:
    print(pipe)


print("\n=== Syncs ===")
for sync in schedule.syncs:
    print(sync)


print("\n=== Lower IR ===")

print("\nBuffers:")
for buffer in program.buffers:
    print(buffer)

print("\nWarp Regions:")
for region in program.warp_regions:
    print(region)

print("\nSyncs:")
for sync in program.syncs:
    print(sync)

print("\n=== Generated CUDA ===")
print(cuda)