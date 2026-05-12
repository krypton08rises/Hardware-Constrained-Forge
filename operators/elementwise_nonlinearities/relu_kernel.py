import torch
import triton
import triton.language as tl


@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Identify which block this program is running
    pid = tl.program_id(axis=0)

    # 2. Calculate the starting memory offset for this block
    block_start = pid * BLOCK_SIZE

    # 3. Create a vector of offsets for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 4. Create a memory mask to prevent out-of-bounds reads
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)


def relu(input: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(input)
    n_elements = input.numel()
    BLOCK_SIZE = 256

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    relu_kernel[grid](input, output, n_elements, BLOCK_SIZE)
    return output


if __name__ == "__main__":
    # time the relu function
    import time

    start = time.perf_counter()
    x = torch.randn(1024, device="cuda")
    y = relu(x)
    print(f"Time taken: {time.perf_counter() - start:.6f} seconds")
    print(y)
    print(torch.allclose(y, torch.nn.functional.relu(x)))
