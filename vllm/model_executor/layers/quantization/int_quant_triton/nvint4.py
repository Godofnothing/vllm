import torch
import triton
import triton.language as tl


@triton.jit
def int_round(x, q_min, q_max):
    return tl.clamp(tl.floor(x + 0.5), q_min, q_max)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 32 * 32}),
        triton.Config({"BLOCK_SIZE": 64 * 32}),
        triton.Config({"BLOCK_SIZE": 128 * 32}),
        triton.Config({"BLOCK_SIZE": 256 * 32}),
        triton.Config({"BLOCK_SIZE": 512 * 32}),
    ],
    key=[],
)
@triton.jit
def nvint4_forward_kernel(
    x_ptr: tl.tensor,
    q_ptr: tl.tensor,
    transform_ptr: tl.tensor,
    global_scale_ptr: tl.tensor,
    N: int,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    APPLY_TRANSFORM: tl.constexpr,
    TRANSFORM_SIZE: tl.constexpr,
):
    # Constants
    INT4_MIN = -7
    INT4_MAX = +7
    UINT8_MAX = 255

    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs, offs < N)
    q = tl.load(q_ptr + offs, offs < N)

    if APPLY_TRANSFORM:
        transform_offs = tl.arange(0, TRANSFORM_SIZE * TRANSFORM_SIZE)
        transform = tl.load(transform_ptr + transform_offs).reshape(TRANSFORM_SIZE, TRANSFORM_SIZE)

        # Apply transform
        x = x.reshape(BLOCK_SIZE // TRANSFORM_SIZE, TRANSFORM_SIZE)
        x_tr = tl.dot(x, transform)
    else:
        x_tr = x

    # Global scale
    global_scale = tl.load(global_scale_ptr)

    # Group
    x_tr = x_tr.reshape(BLOCK_SIZE // GROUP_SIZE, GROUP_SIZE)

    x_tr_abs = x_tr.abs()
    sign = 2 * (x_tr > 0) - 1
    scales = tl.max(x_tr_abs, axis=-1, keep_dims=True)
    scales = global_scale * scales / INT4_MAX
    scales_int8 = (int_round(scales, 0, UINT8_MAX).to(tl.float32) / global_scale).to(x.dtype)

    x_tr_scaled = x_tr_abs / scales_int8

    # Round to FP4 grid
    q = int_round(x_tr_scaled, INT4_MIN, INT4_MAX)

    # Dequantize
    q = q * sign * scales_int8
    # Reshape to original shape
    q = q.reshape(BLOCK_SIZE)
    tl.store(q_ptr + offs, q, offs < N)


def nvint4_forward_kernel_wrapper(
    x: torch.Tensor,
    transform: torch.Tensor,
    global_scale: torch.Tensor,
):
    x_numel = x.numel()
    x_q = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x_numel, meta["BLOCK_SIZE"]),)

    transform_size = transform.shape[0]

    nvint4_forward_kernel[grid](
        x,
        x_q,
        transform,
        global_scale,
        x.numel(),
        GROUP_SIZE=16,
        APPLY_TRANSFORM=True,
        TRANSFORM_SIZE=transform_size,
    )
    return x_q
