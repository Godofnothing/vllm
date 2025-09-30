from itertools import product

import torch
import triton
import triton.language as tl


def get_autotuning_config(configs, named_args, **kwargs):
    hadamard_dim = kwargs["hadamard_dim"]
    # Block size has to be chosen such, that BLOCK_SIZE // hadamard_dim is multiple of 16
    BLOCK_SIZES = [32 * 32, 64 * 32, 128 * 32, 256 * 32, 512 * 32]

    block_size_step = 16 * hadamard_dim

    for config in configs:
        num_warps = config.num_warps
        num_stages = config.num_stages
        for block_size in BLOCK_SIZES:
            if block_size // block_size_step > 0 and block_size % block_size_step == 0:
                yield triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps, num_stages=num_stages) 


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages) 
        # for num_stages, num_warps in product([1, 2], [1, 2, 4, 8])
        for num_stages, num_warps in product([1, 2,], [2, 4])
    ],
    prune_configs_by={"early_config_prune": get_autotuning_config},
    key=[]
)
@triton.jit
def mxfp4_deluxe_forward_kernel(
    x_ptr,
    hadamard_matrix_ptr,
    output_ptr,
    clip_mask_ptr,
    logscales_min_ptr,
    logscales_max_ptr,
    n_elements: tl.constexpr,
    hadamard_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets_hadamard = tl.arange(0, hadamard_dim * hadamard_dim)
    hadamard_matrix = tl.load(hadamard_matrix_ptr + offsets_hadamard).reshape(
        hadamard_dim, hadamard_dim
    )

    # load x
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x_flat = tl.load(x_ptr + offsets, mask=mask)

    # hadamard transform
    x = tl.reshape(x_flat, (BLOCK_SIZE // hadamard_dim, hadamard_dim))
    x_had = tl.dot(x, hadamard_matrix)

    # group
    x_had_grouped = tl.reshape(x_had, (BLOCK_SIZE // 32, 32))

    # load logscale stats
    logscales_min = tl.load(logscales_min_ptr)
    logscales_max = tl.load(logscales_max_ptr)

    # scale
    scales = tl.max(tl.abs(x_had_grouped), axis=-1, keep_dims=True)
    logscales = tl.clamp(tl.log2(scales), min=-128, max=127)
    normalized_logscales = (logscales - logscales_min) / (logscales_max - logscales_min)
    qlogscales = tl.clamp(tl.floor(255.0 * normalized_logscales + 0.5), 0, 255)
    shared_exps = tl.exp2(qlogscales / 255 * (logscales_max - logscales_min) + logscales_min)
    shared_exps = scales / 6
    x_had_scaled = x_had_grouped / shared_exps

    # quantize
    x_had_scaled_abs = tl.abs(x_had_scaled)
    x_had_scaled_sign = tl.where(
        x_had_scaled > 0,
        1,
        -1,
    )

    x_fp4 = (
        tl.where(
            x_had_scaled_abs > 5,
            6,
            tl.where(
                x_had_scaled_abs > 3.5,
                4,
                tl.where(
                    x_had_scaled_abs > 2.5,
                    3,
                    tl.where(
                        x_had_scaled_abs > 1.75,
                        2,
                        tl.where(
                            x_had_scaled_abs > 1.25,
                            1.5,
                            tl.where(
                                x_had_scaled_abs > 0.75,
                                1,
                                tl.where(
                                    x_had_scaled_abs > 0.25,
                                    0.5,
                                    0,
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )
        * x_had_scaled_sign
    )
    if clip_mask_ptr is not None:
        tl.store(
            clip_mask_ptr + offsets,
            tl.reshape(x_had_scaled_abs < 6, (BLOCK_SIZE,)),
            mask=mask,
        )

    # dequantize
    x_dequantized = x_fp4 * shared_exps

    # Reshape back to flat form for storage
    x_dequantized_flat = tl.reshape(x_dequantized, (BLOCK_SIZE,))

    # store
    tl.store(output_ptr + offsets, x_dequantized_flat, mask=mask)


def mxfp4_deluxe_forward_kernel_wrapper(
    x,
    hadamard_matrix,
    logscales_min,
    logscales_max,
    return_clip_mask=False,
):
    # Make sure inputs are contiguous
    x = x.contiguous()

    # Create output tensor
    output = torch.empty_like(x)
    if return_clip_mask:
        clip_mask = torch.empty_like(x, dtype=torch.bool)
    else:
        clip_mask = None

    # Get total number of elements and calculate grid for launching the kernel
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    logscales_max = torch.maximum(logscales_max, logscales_min + 1e-6)

    # Launch optimized kernel
    with torch.cuda.device(x.device):
        mxfp4_deluxe_forward_kernel[grid](
            x_ptr=x,
            hadamard_matrix_ptr=hadamard_matrix,
            output_ptr=output,
            clip_mask_ptr=clip_mask,
            logscales_min_ptr=logscales_min,
            logscales_max_ptr=logscales_max,
            n_elements=n_elements,
            hadamard_dim=hadamard_matrix.shape[-1],
            group_size=32,
        )

    return output, clip_mask
