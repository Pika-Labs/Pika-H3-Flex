// SwiGLU with the LoRA-B up-projection folded in. Same rounding points as
// kernels._swiglu_lora_kernel (= lora_add_ + swiglu of the reference op sequence):
//   la = bf16(t . B[n]) ; lg = bf16(t . B[FF+n]) ; a = bf16(up[m,n] + la) ; g = bf16(up[m,FF+n] + lg)
//   out[m,n] = bf16(a * bf16(g / (1 + exp(-g))))
// t [M, 128] bf16 (row stride st), B [2FF, 128] bf16 contiguous (native LoRA-B layout: k contiguous), up [M, 2FF] (stride su),
// out [M, FF] (stride so).  MFMA v_mfma_f32_16x16x32_bf16 straight from global/L1 (no LDS): lane l holds operand row/col l%16 and
// the 8 reduction indices 32c + 8(l/16) .. +7 (same permutation for both operands -> the full dot over k = 0..127).
// One wave: 64 rows (RS = 4 row sub-tiles of 16) x a column range, 16 columns per step; t fragments stay in registers.
// Block: 4 waves = 256 rows. Grid: (ceil(M / 256), FF / CW).
// Operand / accumulator layout convention: MFMA 16x16 fragments (Frag, f32x4).
#include <hip/hip_runtime.h>
#include <stdint.h>

using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
union Frag { bf16x8 v; u32x4 u; };

__device__ __forceinline__ float bf2f(uint16_t h) { return __uint_as_float(((uint32_t)h) << 16); }
__device__ __forceinline__ uint16_t f2bf(float f) {  // round to nearest even: native v_cvt_pk_bf16_f32 on gfx950
  __bf16 b = (__bf16)f;
  return *reinterpret_cast<uint16_t*>(&b);
}
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
#ifdef FASTDIV
// UNSAFE (measured 2026-11-06): for very negative g, exp(-g) = inf and the correction computes -inf * 0 = NaN. Do not use.
__device__ __forceinline__ float sdiv(float n, float d) {  // rcp + Newton + one correction (not always correctly rounded)
  float r = __builtin_amdgcn_rcpf(d);
  r = __builtin_fmaf(__builtin_fmaf(-d, r, 1.0f), r, r);
  float q = n * r;
  return __builtin_fmaf(__builtin_fmaf(-d, q, n), r, q);
}
#else
__device__ __forceinline__ float sdiv(float n, float d) { return n / d; }
#endif

template <int RS, int CW>
__global__ __launch_bounds__(256) void swiglu_lora_kernel(const uint16_t* __restrict__ up, const uint16_t* __restrict__ t,
                                                          const uint16_t* __restrict__ B, uint16_t* __restrict__ out,
                                                          int M, int FF, int su, int st, int so) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const int lr = lane & 15, lg = lane >> 4;
  const int r0 = (blockIdx.x * 4 + wave) * (16 * RS);
  if (r0 >= M) return;
  const int c0 = blockIdx.y * CW;
  Frag tf[RS][4];
#pragma unroll
  for (int rs = 0; rs < RS; rs++) {
    const int row = r0 + rs * 16 + lr;
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (row < M) tf[rs][c].u = *reinterpret_cast<const u32x4*>(t + (int64_t)row * st + 32 * c + 8 * lg);
      else tf[rs][c].u = u32x4{0, 0, 0, 0};
    }
  }
  for (int n0 = c0; n0 < c0 + CW; n0 += 16) {
    const int n = n0 + lr;
    Frag ba[4], bg[4];
#pragma unroll
    for (int c = 0; c < 4; c++) {
      ba[c].u = *reinterpret_cast<const u32x4*>(B + (int64_t)n * 128 + 32 * c + 8 * lg);
      bg[c].u = *reinterpret_cast<const u32x4*>(B + (int64_t)(FF + n) * 128 + 32 * c + 8 * lg);
    }
    uint16_t ua[RS][4], ug[RS][4];
#pragma unroll
    for (int rs = 0; rs < RS; rs++)
#pragma unroll
      for (int v = 0; v < 4; v++) {
        const int row = r0 + rs * 16 + 4 * lg + v;
        const int64_t off = (int64_t)(row < M ? row : M - 1) * su + n;
        ua[rs][v] = up[off];
        ug[rs][v] = up[off + FF];
      }
#pragma unroll
    for (int rs = 0; rs < RS; rs++) {
      f32x4 aa = {0.f, 0.f, 0.f, 0.f}, ag = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int c = 0; c < 4; c++) {
        aa = __builtin_amdgcn_mfma_f32_16x16x32_bf16(tf[rs][c].v, ba[c].v, aa, 0, 0, 0);
        ag = __builtin_amdgcn_mfma_f32_16x16x32_bf16(tf[rs][c].v, bg[c].v, ag, 0, 0, 0);
      }
#pragma unroll
      for (int v = 0; v < 4; v++) {
        const int row = r0 + rs * 16 + 4 * lg + v;
        const float a = rbf(bf2f(ua[rs][v]) + rbf(aa[v]));
        const float g = rbf(bf2f(ug[rs][v]) + rbf(ag[v]));
        const float sg = rbf(sdiv(g, 1.0f + expf(-g)));
        if (row < M) out[(int64_t)row * so + n] = f2bf(a * sg);
      }
    }
  }
}

// v2: transposed MFMA (A operand = LoRA-B rows, B operand = t rows): D^T[n][m]. A-operand row i of n-sub-tile ns is column
// n0 + 16 (i / 4) + 4 ns + (i % 4), so output lane l (g = l / 16, m = l % 16) owns the 16 CONTIGUOUS columns n0 + 16 g .. +15 of
// row m over the 4 n-sub-tiles -> 32-byte vector loads of up / stores of out. A wave: RS x 16 rows, 64 columns per step.
using u16x16 = uint16_t __attribute__((ext_vector_type(16)));
template <int RS, int CW>
__global__ __launch_bounds__(256) void swiglu_lora_v2(const uint16_t* __restrict__ up, const uint16_t* __restrict__ t,
                                                      const uint16_t* __restrict__ B, uint16_t* __restrict__ out,
                                                      int M, int FF, int su, int st, int so) {
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const int lr = lane & 15, lg = lane >> 4;
  const int r0 = (blockIdx.x * 4 + wave) * (16 * RS);
  if (r0 >= M) return;
  const int c0 = blockIdx.y * CW;
  Frag tf[RS][4];
#pragma unroll
  for (int rs = 0; rs < RS; rs++) {
    const int row = r0 + rs * 16 + lr;
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (row < M) tf[rs][c].u = *reinterpret_cast<const u32x4*>(t + (int64_t)row * st + 32 * c + 8 * lg);
      else tf[rs][c].u = u32x4{0, 0, 0, 0};
    }
  }
  const int ncol_a = 16 * (lr >> 2) + (lr & 3);  // A-operand row lr of n-sub-tile ns -> column n0 + ncol_a + 4 ns
  for (int n0 = c0; n0 < c0 + CW; n0 += 64) {
    u16x16 ua[RS], ug[RS];
#pragma unroll
    for (int rs = 0; rs < RS; rs++) {
      const int row = r0 + rs * 16 + lr;
      const int64_t off = (int64_t)(row < M ? row : M - 1) * su + n0 + 16 * lg;
      ua[rs] = *reinterpret_cast<const u16x16*>(up + off);
      ug[rs] = *reinterpret_cast<const u16x16*>(up + off + FF);
    }
    f32x4 aa[RS][4], ag[RS][4];
#pragma unroll
    for (int ns = 0; ns < 4; ns++) {
      const int n = n0 + ncol_a + 4 * ns;
      Frag ba[4], bg[4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        ba[c].u = *reinterpret_cast<const u32x4*>(B + (int64_t)n * 128 + 32 * c + 8 * lg);
        bg[c].u = *reinterpret_cast<const u32x4*>(B + (int64_t)(FF + n) * 128 + 32 * c + 8 * lg);
      }
#pragma unroll
      for (int rs = 0; rs < RS; rs++) {
        aa[rs][ns] = f32x4{0.f, 0.f, 0.f, 0.f}; ag[rs][ns] = f32x4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int c = 0; c < 4; c++) {
          aa[rs][ns] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(ba[c].v, tf[rs][c].v, aa[rs][ns], 0, 0, 0);
          ag[rs][ns] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(bg[c].v, tf[rs][c].v, ag[rs][ns], 0, 0, 0);
        }
      }
    }
#pragma unroll
    for (int rs = 0; rs < RS; rs++) {
      const int row = r0 + rs * 16 + lr;
      u16x16 o;
#pragma unroll
      for (int ns = 0; ns < 4; ns++)
#pragma unroll
        for (int v = 0; v < 4; v++) {
          const int e = 4 * ns + v;
          const float a = rbf(bf2f(ua[rs][e]) + rbf(aa[rs][ns][v]));
          const float g = rbf(bf2f(ug[rs][e]) + rbf(ag[rs][ns][v]));
          const float sg = rbf(sdiv(g, 1.0f + expf(-g)));
          o[e] = f2bf(a * sg);
        }
      if (row < M) *reinterpret_cast<u16x16*>(out + (int64_t)row * so + n0 + 16 * lg) = o;
    }
  }
}

// v3: v2 + LoRA-B tiles staged in LDS once per block (4 waves x RS x 16 rows share them), double-buffered: the global loads of
// tile s+1 are issued before the MFMAs of tile s. LDS row stride 136 bf16 (272 B) keeps the 16-row fragment reads conflict-free.
constexpr int kLdsRow = 136;
template <int RS, int CW>
__global__ __launch_bounds__(256) void swiglu_lora_v3(const uint16_t* __restrict__ up, const uint16_t* __restrict__ t,
                                                      const uint16_t* __restrict__ B, uint16_t* __restrict__ out,
                                                      int M, int FF, int su, int st, int so) {
  __shared__ __attribute__((aligned(16))) uint16_t sb[2][128][kLdsRow];  // [buf][a rows 0..63 | g rows 64..127][k]
  const int tid = threadIdx.x, lane = tid & 63, wave = tid >> 6;
  const int lr = lane & 15, lg = lane >> 4;
  const int r0 = (blockIdx.x * 4 + wave) * (16 * RS);
  const int c0 = blockIdx.y * CW;
  // cooperative tile load: 128 rows x 256 B = 2048 x 16 B units, 8 per thread; unit u: row u / 16, k8 = u % 16
  u32x4 pf[8];
  auto gload = [&](int n0) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int u = tid + 256 * j, r = u >> 4, k8 = u & 15;
      const int64_t brow = (r < 64) ? (n0 + r) : (FF + n0 + r - 64);
      pf[j] = *reinterpret_cast<const u32x4*>(B + brow * 128 + 8 * k8);
    }
  };
  auto lstore = [&](int buf) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int u = tid + 256 * j, r = u >> 4, k8 = u & 15;
      *reinterpret_cast<u32x4*>(&sb[buf][r][8 * k8]) = pf[j];
    }
  };
  Frag tf[RS][4];
#pragma unroll
  for (int rs = 0; rs < RS; rs++) {
    const int row = r0 + rs * 16 + lr;
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (row < M) tf[rs][c].u = *reinterpret_cast<const u32x4*>(t + (int64_t)row * st + 32 * c + 8 * lg);
      else tf[rs][c].u = u32x4{0, 0, 0, 0};
    }
  }
  const int ncol_a = 16 * (lr >> 2) + (lr & 3);
  gload(c0);
  lstore(0);
  __syncthreads();
  constexpr int NSTEP = CW / 64;
  for (int sidx = 0; sidx < NSTEP; sidx++) {
    const int n0 = c0 + 64 * sidx, buf = sidx & 1;
    if (sidx + 1 < NSTEP) gload(n0 + 64);
    u16x16 ua[RS], ug[RS];
#pragma unroll
    for (int rs = 0; rs < RS; rs++) {
      const int row = r0 + rs * 16 + lr;
      const int64_t off = (int64_t)(row < M ? row : M - 1) * su + n0 + 16 * lg;
      ua[rs] = *reinterpret_cast<const u16x16*>(up + off);
      ug[rs] = *reinterpret_cast<const u16x16*>(up + off + FF);
    }
    f32x4 aa[RS][4], ag[RS][4];
#pragma unroll
    for (int ns = 0; ns < 4; ns++) {
      const int nl = ncol_a + 4 * ns;
      Frag ba[4], bg[4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        ba[c].u = *reinterpret_cast<const u32x4*>(&sb[buf][nl][32 * c + 8 * lg]);
        bg[c].u = *reinterpret_cast<const u32x4*>(&sb[buf][64 + nl][32 * c + 8 * lg]);
      }
#pragma unroll
      for (int rs = 0; rs < RS; rs++) {
        aa[rs][ns] = f32x4{0.f, 0.f, 0.f, 0.f}; ag[rs][ns] = f32x4{0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int c = 0; c < 4; c++) {
          aa[rs][ns] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(ba[c].v, tf[rs][c].v, aa[rs][ns], 0, 0, 0);
          ag[rs][ns] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(bg[c].v, tf[rs][c].v, ag[rs][ns], 0, 0, 0);
        }
      }
    }
#pragma unroll
    for (int rs = 0; rs < RS; rs++) {
      const int row = r0 + rs * 16 + lr;
      u16x16 o;
#pragma unroll
      for (int ns = 0; ns < 4; ns++)
#pragma unroll
        for (int v = 0; v < 4; v++) {
          const int e = 4 * ns + v;
          const float a = rbf(bf2f(ua[rs][e]) + rbf(aa[rs][ns][v]));
          const float g = rbf(bf2f(ug[rs][e]) + rbf(ag[rs][ns][v]));
          const float sg = rbf(sdiv(g, 1.0f + expf(-g)));
          o[e] = f2bf(a * sg);
        }
      if (row < M) *reinterpret_cast<u16x16*>(out + (int64_t)row * so + n0 + 16 * lg) = o;
    }
    if (sidx + 1 < NSTEP) lstore(buf ^ 1);
    __syncthreads();
  }
}

extern "C" int swiglu_lora_launch(const void* up, const void* t, const void* B, void* out, int M, int FF, int su, int st, int so,
                                  int cw, hipStream_t stream) {
  dim3 grid((M + 255) / 256, FF / cw), block(256);
  if (cw == 512) hipLaunchKernelGGL((swiglu_lora_kernel<4, 512>), grid, block, 0, stream, (const uint16_t*)up, (const uint16_t*)t,
                                    (const uint16_t*)B, (uint16_t*)out, M, FF, su, st, so);
  else if (cw == 1024) hipLaunchKernelGGL((swiglu_lora_kernel<4, 1024>), grid, block, 0, stream, (const uint16_t*)up, (const uint16_t*)t,
                                          (const uint16_t*)B, (uint16_t*)out, M, FF, su, st, so);
  else if (cw == 2048) hipLaunchKernelGGL((swiglu_lora_kernel<4, 2048>), grid, block, 0, stream, (const uint16_t*)up, (const uint16_t*)t,
                                          (const uint16_t*)B, (uint16_t*)out, M, FF, su, st, so);
  else return -1;
  return (int)hipGetLastError();
}

extern "C" int swiglu_lora_v2_launch(const void* up, const void* t, const void* B, void* out, int M, int FF, int su, int st, int so,
                                     int cw, int rs, hipStream_t stream) {
  dim3 block(256);
#define L2(RS_, CW_) if (rs == RS_ && cw == CW_) { dim3 grid((M + 64 * RS_ - 1) / (64 * RS_), FF / CW_); \
    hipLaunchKernelGGL((swiglu_lora_v2<RS_, CW_>), grid, block, 0, stream, (const uint16_t*)up, (const uint16_t*)t, (const uint16_t*)B, \
                       (uint16_t*)out, M, FF, su, st, so); return (int)hipGetLastError(); }
  L2(1, 512) L2(1, 1024) L2(1, 2048) L2(2, 512) L2(2, 1024) L2(2, 2048) L2(4, 512) L2(4, 1024)
#undef L2
  return -1;
}

extern "C" int swiglu_lora_v3_launch(const void* up, const void* t, const void* B, void* out, int M, int FF, int su, int st, int so,
                                     int cw, int rs, hipStream_t stream) {
  dim3 block(256);
#define L3(RS_, CW_) if (rs == RS_ && cw == CW_) { dim3 grid((M + 64 * RS_ - 1) / (64 * RS_), FF / CW_); \
    hipLaunchKernelGGL((swiglu_lora_v3<RS_, CW_>), grid, block, 0, stream, (const uint16_t*)up, (const uint16_t*)t, (const uint16_t*)B, \
                       (uint16_t*)out, M, FF, su, st, so); return (int)hipGetLastError(); }
  L3(1, 512) L3(1, 1024) L3(1, 2048) L3(2, 512) L3(2, 1024) L3(2, 2048) L3(4, 512) L3(4, 1024) L3(4, 2048)
#undef L3
  return -1;
}
