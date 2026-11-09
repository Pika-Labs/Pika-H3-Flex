// SPDX-License-Identifier: Apache-2.0
// host shim for the asm kernels fa17 / fa18 (code object loaded at first call, path = this .so with .co, or $FA17_CO).
// fa6_launch: same C ABI as fa16.hip (Nq = Nk = N). fa18_launch: separate query rows Nq and key rows N.
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string>
struct Fa16Args {
  const void* q; const void* k; const void* vt; void* o; float* lse;
  long long sq_s, sq_h, sk_s, sk_h, svt_h, svt_d, so_s, so_h;
  int N; float c; float scale; int Nq;
};
static hipFunction_t g_fn = nullptr;
static std::string co_path() {
  const char* e = getenv("FA17_CO");
  if (e) return e;
  Dl_info info; dladdr((void*)&co_path, &info);
  std::string p = info.dli_fname; size_t d = p.rfind(".so");
  return p.substr(0, d) + ".co";
}
extern "C" int fa18_launch(const void* q, const void* k, const void* vt, void* o, float* lse, long long sq_s, long long sq_h,
                           long long sk_s, long long sk_h, long long svt_h, long long svt_d, long long so_s, long long so_h,
                           int Nq, int N, int H, float scale, hipStream_t st) {
  if (!g_fn) {
    hipModule_t m; std::string p = co_path();
    if (hipModuleLoad(&m, p.c_str()) != hipSuccess) { fprintf(stderr, "fa17: cannot load %s\n", p.c_str()); return -1; }
    if (hipModuleGetFunction(&g_fn, m, "fa17_fwd") != hipSuccess) { fprintf(stderr, "fa17: no fa17_fwd\n"); return -2; }
  }
  Fa16Args a;
  a.q = q; a.k = k; a.vt = vt; a.o = o; a.lse = lse;
  a.sq_s = sq_s; a.sq_h = sq_h; a.sk_s = sk_s; a.sk_h = sk_h; a.svt_h = svt_h; a.svt_d = svt_d; a.so_s = so_s; a.so_h = so_h;
  a.N = N; a.scale = scale; a.c = scale * 1.4426950408889634f; a.Nq = Nq;
  size_t sz = sizeof(a);
  void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
  return (int)hipModuleLaunchKernel(g_fn, (Nq + 255) / 256, H, 1, 512, 1, 1, 0, st, nullptr, cfg);
}
extern "C" int fa6_launch(const void* q, const void* k, const void* vt, void* o, float* lse, long long sq_s, long long sq_h,
                          long long sk_s, long long sk_h, long long svt_h, long long svt_d, long long so_s, long long so_h,
                          int N, int H, float scale, hipStream_t st) {
  return fa18_launch(q, k, vt, o, lse, sq_s, sq_h, sk_s, sk_h, svt_h, svt_d, so_s, so_h, N, N, H, scale, st);
}
