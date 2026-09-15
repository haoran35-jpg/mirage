from twill_importer import import_twill
from passes import verify_schedule, insert_sync
from lowering import lower_ws_to_mpk
from moe_linear_codegen import (
    write_moe_linear,
    ampere_template_path,
    ampere_output_path,
)


# Hopper/Blackwell MoE (moe_linear_swapAB_hopper.cuh):
#   warps >= 4  DMA warpgroup — TMA A + cp.async B + mbarrier publish
#   warps <  4  MMA warpgroup — wait ring, wgmma, release, epilogue
#   warp 0      init a_full / b_full / ab_empty rings
opw = {
    "init_mbar": 0,
    "tma_A": 4,
    "cpasync_B": 4,
    "mma": 0,
    "epilogue": 0,
}

# 8-wide overlap → NUM_AB_STAGE = 8 (Hopper mbarrier ring depth).
liveness = {
    "tma_A": {i: list(range(i, i + 8)) for i in range(8)},
    "cpasync_B": {i: list(range(i, i + 8)) for i in range(8)},
}

pipeline_info = {
    "tma_A": (
        "A_pipeline",
        "sA",
        "mma",
    ),
    "cpasync_B": (
        "B_pipeline",
        "sB",
        "mma",
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

# 5. Codegen into generated/moe_linear_swapAB_hopper.cuh
output = write_moe_linear(schedule, program)


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

print(f"\nk_stage = {program.k_stage}")
print(f"\n=== Wrote {output} ===")

generated = output.read_text()
assert "#define TWILL_NUM_AB_STAGE 8" in generated
assert "#define TWILL_DMA_WARP 4" in generated
assert "#define TWILL_INIT_WARP 0" in generated
assert "if (warp_idx >= TWILL_DMA_WARP)" in generated
assert "else if (warp_idx < TWILL_DMA_WARP)" in generated
assert "[twill:tma_A]" in generated
assert "[twill:cpasync_B]" in generated
assert "[twill:mma]" in generated
assert "[twill:epilogue]" in generated
assert "[twill:mbar_a_full]" in generated
assert "[twill:mbar_b_full]" in generated
assert "[twill:mbar_ab_empty]" in generated
assert "Twill dma role (warp_idx >= 4): tma_A@w4, cpasync_B@w4" in generated
assert "sync A_pipeline: warp 4 -> warp 0" in generated
print("Hopper role-if + mbarrier ring + TMA/cp.async predicates present")

# Ampere WS: same Hopper role split, cp.async both A and B, software ring.
ampere_opw = {
    "init_mbar": 0,
    "g2s_A": 4,
    "g2s_B": 4,
    "s2r": 0,
    "mma": 0,
    "epilogue": 0,
}
ampere_liveness = {
    "g2s_A": {i: list(range(i, i + 3)) for i in range(3)},
    "g2s_B": {i: list(range(i, i + 3)) for i in range(3)},
}
ampere_pipeline_info = {
    "g2s_A": ("A_pipeline", "Ashm", "mma"),
    "g2s_B": ("B_pipeline", "Bshm", "mma"),
}
ampere_schedule = insert_sync(
    import_twill(ampere_opw, ampere_liveness, ampere_pipeline_info)
)
verify_schedule(ampere_schedule)
ampere_program = lower_ws_to_mpk(ampere_schedule)
ampere_out = write_moe_linear(
    ampere_schedule,
    ampere_program,
    output_path=ampere_output_path(),
    template_path=ampere_template_path(),
)
ampere_cuda = ampere_out.read_text()
assert "#define TWILL_PIPE_MAX 3" in ampere_cuda
assert "#define TWILL_DMA_WARP 4" in ampere_cuda
assert "if (warp_idx >= TWILL_DMA_WARP)" in ampere_cuda
assert "[twill:g2s_A]" in ampere_cuda
assert "[twill:g2s_B]" in ampere_cuda
assert "[twill:mma]" in ampere_cuda
assert "[twill:mbar_ab_full]" in ampere_cuda
assert "[twill:mbar_ab_empty]" in ampere_cuda
assert "ampere_wait_cnt" in ampere_cuda
assert "MOE_NUM_THREADS 256" in ampere_cuda
print(f"=== Wrote {ampere_out} ===")
print("Ampere Hopper-style WS + software ring + cp.async present")
