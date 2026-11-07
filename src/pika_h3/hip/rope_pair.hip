// qk_rope_pack_lora for pair-interleaved q/k head dims. Same contract and rounding points as
// kernels._qk_rope_pair_kernel (PIKA_ROPE_PAIR):
//   lo = bf16(t_part . B[col]) ; y' = bf16(y + lo) ; v: out = y'
//   q/k: rstd = 1 / sqrt(sum(y'^2) / 128 + eps) ; n = bf16(y' * rstd * w) ; pairs (2p, 2p+1), p < RD/2:
//        a = bf16(bf16(n_a * cos[p]) + bf16(-n_b * sin[p])) ; b = bf16(bf16(n_b * cos[HALF+p]) + bf16(n_a * sin[HALF+p])) ; else n.
// The LoRA term is computed transposed, lo^T[hd, tok] = B_h[hd, k] t^T[k, tok], with v_mfma_f32_16x16x32_bf16: the accumulator of
// lane l holds head dims 16T + 4(l/16) + i (i = 0..3) of token l % 16, i.e. 4 contiguous head dims = one 8-byte load / store, and the
// RoPE pair partner sits in the same lane. No LDS, no layout change between the dot and the elementwise part.
// One wave: 16 * RS tokens x 1 head; block 4 waves; grid (ceil(rows / (64 RS)), NH).
#include <hip/hip_runtime.h>
#include <stdint.h>

using bf16x8 = __bf16 __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));
using u32x4 = uint32_t __attribute__((ext_vector_type(4)));
using u32x2 = uint32_t __attribute__((ext_vector_type(2)));
union Frag { bf16x8 v; u32x4 u; };

__device__ __forceinline__ float bf2f(uint32_t h) { return __uint_as_float(h << 16); }
__device__ __forceinline__ uint32_t f2bf(float f) {
  __bf16 b = (__bf16)f;
  return (uint32_t)(*reinterpret_cast<uint16_t*>(&b));
}
__device__ __forceinline__ float rbf(float f) { return bf2f(f2bf(f)); }
__device__ __forceinline__ float lo16(uint32_t w) { return bf2f(w & 0xffffu); }
__device__ __forceinline__ float hi16(uint32_t w) { return bf2f(w >> 16); }
__device__ __forceinline__ uint32_t pack(float a, float b) { return f2bf(a) | (f2bf(b) << 16); }

template <int RS, bool SPLIT, int NW>
__global__ __launch_bounds__(64 * NW) void rope_pair_kernel(const uint16_t* __restrict__ qkv, const uint16_t* __restrict__ t,
                                                        const uint16_t* __restrict__ B, const uint16_t* __restrict__ qn,
                                                        const uint16_t* __restrict__ kn, const uint16_t* __restrict__ cosp,
                                                        const uint16_t* __restrict__ sinp, uint16_t* __restrict__ out, int s,
                                                        float eps, int st, int sin_, int s0, int row0, int row1, int NH, int HPR,
                                                        int RD) {
  constexpr int HD = 128, R = 128;
  const int lane = threadIdx.x & 63, wave = threadIdx.x >> 6;
  const int lr = lane & 15, lg = lane >> 4;
  const int head = blockIdx.y;
  const int tb = row0 + (blockIdx.x * NW + wave) * (16 * RS);
  constexpr int PITCH = HD + 8;  // u16 per LDS row (272 B: staggers the banks)
  __shared__ __attribute__((aligned(16))) uint16_t lds_all[NW][16 * RS * PITCH];
  uint16_t* lds = lds_all[wave];
  __shared__ __attribute__((aligned(16))) uint16_t bl[HD * PITCH];  // LoRA-B rows of this head / part, shared by the 4 waves
  const int g = head / HPR, hl = head % HPR;
  const int HALF = RD / 2;
  int64_t dst[RS];
  bool ok[RS];
#pragma unroll
  for (int rs = 0; rs < RS; rs++) {
    const int64_t tok = tb + rs * 16 + lr;
    ok[rs] = tok < row1;
    int64_t drow;
    if (SPLIT) {
      const int NG = NH / HPR;
      drow = tok < s0 ? (int64_t)g * s0 + tok : (int64_t)NG * s0 + (int64_t)g * (s - s0) + (tok - s0);
    } else {
      drow = (int64_t)g * s + tok;
    }
    dst[rs] = drow * 3 * (HPR * HD) + hl * HD + 4 * lg;
  }
  auto drow_of = [&](int64_t tok) -> int64_t {
    if (SPLIT) {
      const int NG = NH / HPR;
      return tok < s0 ? (int64_t)g * s0 + tok : (int64_t)NG * s0 + (int64_t)g * (s - s0) + (tok - s0);
    }
    return (int64_t)g * s + tok;
  };
#pragma unroll 1
  for (int which = 0; which < 3; which++) {
    const int col0 = which * NH * HD + head * HD;
    Frag tf[RS][4];
#pragma unroll
    for (int rs = 0; rs < RS; rs++) {
      const int64_t ri = tb + rs * 16 + lr - row0;
#pragma unroll
      for (int c = 0; c < 4; c++)
        tf[rs][c].u = ok[rs] ? *reinterpret_cast<const u32x4*>(t + ri * st + which * R + 32 * c + 8 * lg) : u32x4{0, 0, 0, 0};
    }
    u32x2 yv[RS][8];  // y' (bf16 packed) for head dims 16T + 4lg .. +3
#pragma unroll
    for (int it = 0; it < 32 / NW; it++) {  // 128 rows x 256 B = 32 KB, 16 B per thread
      const int ch = it * (64 * NW) + threadIdx.x, rw = ch >> 4, pc = ch & 15;
      *reinterpret_cast<u32x4*>(bl + rw * PITCH + 8 * pc) = *reinterpret_cast<const u32x4*>(B + (int64_t)(col0 + rw) * R + 8 * pc);
    }
    // coalesced global -> LDS (16 B per lane, 16 lanes per 256-B head row), then 8-B reads in the MFMA layout
#pragma unroll
    for (int it = 0; it < 4 * RS; it++) {
      const int ch = it * 64 + lane, tk = ch >> 4, pc = ch & 15;
      const int64_t tok = tb + tk;
      u32x4 v = u32x4{0, 0, 0, 0};
      if (tok < row1) v = __builtin_nontemporal_load(reinterpret_cast<const u32x4*>(qkv + (tok - row0) * sin_ + col0 + 8 * pc));
      *reinterpret_cast<u32x4*>(lds + tk * PITCH + 8 * pc) = v;
    }
    __syncthreads();
#pragma unroll
    for (int rs = 0; rs < RS; rs++)
#pragma unroll
      for (int T = 0; T < 8; T++) yv[rs][T] = *reinterpret_cast<const u32x2*>(lds + (16 * rs + lr) * PITCH + 16 * T + 4 * lg);
#pragma unroll
    for (int T = 0; T < 8; T++) {
      Frag ba[4];
#pragma unroll
      for (int c = 0; c < 4; c++) ba[c].u = *reinterpret_cast<const u32x4*>(bl + (16 * T + lr) * PITCH + 32 * c + 8 * lg);
#pragma unroll
      for (int rs = 0; rs < RS; rs++) {
        f32x4 acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int c = 0; c < 4; c++) acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(ba[c].v, tf[rs][c].v, acc, 0, 0, 0);
        const u32x2 y = yv[rs][T];
        const float y0 = rbf(lo16(y.x) + rbf(acc[0])), y1 = rbf(hi16(y.x) + rbf(acc[1]));
        const float y2 = rbf(lo16(y.y) + rbf(acc[2])), y3 = rbf(hi16(y.y) + rbf(acc[3]));
        yv[rs][T] = u32x2{pack(y0, y1), pack(y2, y3)};
      }
    }
    if (which == 2) {
#pragma unroll
      for (int rs = 0; rs < RS; rs++)
#pragma unroll
        for (int T = 0; T < 8; T++) *reinterpret_cast<u32x2*>(lds + (16 * rs + lr) * PITCH + 16 * T + 4 * lg) = yv[rs][T];
    } else {
    const uint16_t* wp = which == 0 ? qn : kn;
#pragma unroll
    for (int rs = 0; rs < RS; rs++) {
      float ss = 0.f;
#pragma unroll
      for (int T = 0; T < 8; T++) {
        const float a = lo16(yv[rs][T].x), b = hi16(yv[rs][T].x), c = lo16(yv[rs][T].y), d = hi16(yv[rs][T].y);
        ss += a * a + b * b + c * c + d * d;
      }
      ss += __shfl_xor(ss, 16);
      ss += __shfl_xor(ss, 32);
      const float rstd = 1.0f / sqrtf(ss / HD + eps);
      const int64_t tok = tb + rs * 16 + lr;
#pragma unroll
      for (int T = 0; T < 8; T++) {
        const u32x2 wv = *reinterpret_cast<const u32x2*>(wp + 16 * T + 4 * lg);
        float n[4];
        n[0] = rbf(lo16(yv[rs][T].x) * rstd * lo16(wv.x));
        n[1] = rbf(hi16(yv[rs][T].x) * rstd * hi16(wv.x));
        n[2] = rbf(lo16(yv[rs][T].y) * rstd * lo16(wv.y));
        n[3] = rbf(hi16(yv[rs][T].y) * rstd * hi16(wv.y));
        const int p = 8 * T + 2 * lg;  // pairs p, p + 1
        float r[4];
        if (p < HALF && ok[rs]) {
          const uint32_t ca = *reinterpret_cast<const uint32_t*>(cosp + tok * RD + p);
          const uint32_t sa = *reinterpret_cast<const uint32_t*>(sinp + tok * RD + p);
          const uint32_t cb = *reinterpret_cast<const uint32_t*>(cosp + tok * RD + HALF + p);
          const uint32_t sb = *reinterpret_cast<const uint32_t*>(sinp + tok * RD + HALF + p);
#pragma unroll
          for (int j = 0; j < 2; j++) {
            const float c_a = j ? hi16(ca) : lo16(ca), s_a = j ? hi16(sa) : lo16(sa);
            const float c_b = j ? hi16(cb) : lo16(cb), s_b = j ? hi16(sb) : lo16(sb);
            const float na = n[2 * j], nb = n[2 * j + 1];
            r[2 * j] = rbf(rbf(na * c_a) + rbf(-nb * s_a));
            r[2 * j + 1] = rbf(rbf(nb * c_b) + rbf(na * s_b));
          }
        } else {
#pragma unroll
          for (int j = 0; j < 4; j++) r[j] = n[j];
        }
        *reinterpret_cast<u32x2*>(lds + (16 * rs + lr) * PITCH + 16 * T + 4 * lg) = u32x2{pack(r[0], r[1]), pack(r[2], r[3])};
      }
    }
    }
    __syncthreads();
#pragma unroll
    for (int it = 0; it < 4 * RS; it++) {
      const int ch = it * 64 + lane, tk = ch >> 4, pc = ch & 15;
      const int64_t tok = tb + tk;
      if (tok < row1)
        __builtin_nontemporal_store(*reinterpret_cast<const u32x4*>(lds + tk * PITCH + 8 * pc),
                                    reinterpret_cast<u32x4*>(out + drow_of(tok) * 3 * (HPR * HD) + hl * HD + which * (HPR * HD) + 8 * pc));
    }
    __syncthreads();
  }
}

extern "C" int rope_pair_launch(const void* qkv, const void* t, const void* B, const void* qn, const void* kn, const void* cosp,
                                const void* sinp, void* out, int s, float eps, int st, int sin_, int s0, int split, int row0,
                                int row1, int NH, int HPR, int RD, int rs, hipStream_t stream) {
  const int rows = row1 - row0;
  const int rsv = rs % 10, nw = rs >= 10 ? 8 : 4;   // rs = RS (4 waves) or 10 + RS (8 waves)
  const int per = 16 * nw * rsv;
  dim3 grid((rows + per - 1) / per, NH), block(64 * nw);
#define L(RS_, SP_, NW_)                                                                                                   \
  hipLaunchKernelGGL((rope_pair_kernel<RS_, SP_, NW_>), grid, block, 0, stream, (const uint16_t*)qkv, (const uint16_t*)t, \
                     (const uint16_t*)B, (const uint16_t*)qn, (const uint16_t*)kn, (const uint16_t*)cosp,                    \
                     (const uint16_t*)sinp, (uint16_t*)out, s, eps, st, sin_, s0, row0, row1, NH, HPR, RD)
#define LS(RS_, NW_) { if (split) L(RS_, true, NW_); else L(RS_, false, NW_); }
  if (rsv == 1 && nw == 4) LS(1, 4)
  else if (rsv == 2 && nw == 4) LS(2, 4)
  else if (rsv == 1 && nw == 8) LS(1, 8)
  else if (rsv == 2 && nw == 8) LS(2, 8)
  else return -1;
#undef LS
#undef L
  return (int)hipGetLastError();
}
