import pytest
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor, block_size: int, num_warps: int):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size, num_warps=num_warps)
    return output


@pytest.mark.parametrize("size", [127, 95532])
@pytest.mark.parametrize("block_size", [128, 1024])
@pytest.mark.parametrize("num_warps", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_vector_add(size, block_size, num_warps, dtype):
    torch.manual_seed(0)
    x = torch.rand(size, device=DEVICE, dtype=dtype)
    y = torch.rand(size, device=DEVICE, dtype=dtype)

    output_torch = x + y
    output_triton = add(x, y, block_size, num_warps)
    max_diff = torch.max(torch.abs(output_torch - output_triton)).item()
    assert max_diff == 0, (f"size={size}, block_size={block_size}, num_warps={num_warps}, max_diff={max_diff}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
