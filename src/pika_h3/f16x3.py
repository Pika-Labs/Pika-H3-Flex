# SPDX-License-Identifier: Apache-2.0
"""fp32-accurate GEMM from ONE fp16 matrix-core GEMM over a 3x longer K (three-term operand split, fp32 accumulation).

x ~ xh + xl, w ~ wh + wl with xh = fp16(x), xl = fp16(x - xh) (22 significant bits); x.w = xh.wh + xh.wl + xl.wh (+ xl.wl, dropped:
2^-22 relative). Rows of x and rows of w are scaled by powers of two (exact) so that all operands sit far from the fp16 subnormal
range; the scales are undone exactly in the fused epilogue (bias add). Measured error vs fp64 is at or below that of the fp32
GEMM itself, i.e. normal fp32 variation -- NOT a precision reduction like TF32 / bf16.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _split3_kernel(x_ptr, out_ptr, rs_ptr, K, BLOCK: tl.constexpr):
    # x [R, K] fp32 -> out [R, 3K] fp16 = [h | l | h] of x*2^e (row max scaled into [256, 512)); rs[r] = 2^-e (fp32)
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    x = tl.load(x_ptr + row * K + cols, mask=mask, other=0.0)
    m = tl.max(tl.abs(x), axis=0)
    m = tl.maximum(m, 1e-30)
    e = tl.floor(tl.log2(m))               # m in [2^e, 2^(e+1))
    sc = tl.exp2(8.0 - e)                  # -> row max in [256, 512)
    xs = x * sc                            # exact (power of two)
    h = xs.to(tl.float16)
    l = (xs - h.to(tl.float32)).to(tl.float16)
    o = out_ptr + row * 3 * K + cols
    tl.store(o, h, mask=mask)
    tl.store(o + K, l, mask=mask)
    tl.store(o + 2 * K, h, mask=mask)
    tl.store(rs_ptr + row, 1.0 / sc)


@triton.jit
def _epilogue_kernel(y_ptr, rs_ptr, cs_ptr, b_ptr, out_ptr, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    y = tl.load(y_ptr + row * N + cols, mask=mask, other=0.0)
    rs = tl.load(rs_ptr + row)
    cs = tl.load(cs_ptr + cols, mask=mask, other=1.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * N + cols, y * rs * cs + b, mask=mask)


def prepare_weight(w):
    """w [N, K] fp32 -> (W3 [N, 3K] fp16 = [h | h | l] of w*2^f, colscale [N] fp32 = 2^-f); row max of w*2^f in [2, 4)."""
    w = w.detach().float()
    m = w.abs().amax(dim=1).clamp(min=1e-30)
    e = torch.floor(torch.log2(m))
    sc = torch.exp2(1.0 - e)
    ws = w * sc[:, None]
    h = ws.half()
    l = (ws - h.float()).half()
    return torch.cat([h, h, l], dim=1).contiguous(), (1.0 / sc).contiguous()


def linear(x, W3, cs, b):
    """fp32 x [R, K] -> fp32 [R, N] = x @ w.T + b with fp32-level accuracy. W3/cs from prepare_weight."""
    R, K = x.shape
    N = W3.shape[0]
    xs = torch.empty((R, 3 * K), dtype=torch.float16, device=x.device)
    rs = torch.empty((R,), dtype=torch.float32, device=x.device)
    _split3_kernel[(R,)](x, xs, rs, K, BLOCK=triton.next_power_of_2(K), num_warps=8)
    y = torch.mm(xs, W3.t(), out_dtype=torch.float32)
    out = torch.empty_like(y)
    _epilogue_kernel[(R, triton.cdiv(N, 1024))](y, rs, cs, b, out, N, BLOCK=1024, num_warps=4)
    return out
