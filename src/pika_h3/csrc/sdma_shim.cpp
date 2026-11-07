// SDMA all-to-all shim: HSA engine copies + HIP stream write/wait values. Loaded by ctypes from each rank process.
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
struct WaitArgs { uint64_t addr[16]; int64_t val[16]; int32_t eq[16]; int32_t n; };
// one thread spins until every condition holds: *addr >= val (eq = 0) or *addr == val (eq = 1); system-scope loads
__global__ void pika_wait_kernel(WaitArgs a) {
  if (threadIdx.x != 0) return;
  for (int i = 0; i < a.n; ++i) {
    int64_t* p = (int64_t*)a.addr[i];
    if (a.eq[i]) { while (__hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != a.val[i]) __builtin_amdgcn_s_sleep(2); }
    else { while (__hip_atomic_load(p, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < a.val[i]) __builtin_amdgcn_s_sleep(2); }
  }
}
extern "C" {
int sd_wait_kernel(void* stream, int n, const uint64_t* addr, const int64_t* val, const int32_t* eq) {
  WaitArgs a; memset(&a, 0, sizeof(a)); if (n > 16) return -1; a.n = n;
  for (int i = 0; i < n; ++i) { a.addr[i] = addr[i]; a.val[i] = val[i]; a.eq[i] = eq[i]; }
  hipLaunchKernelGGL(pika_wait_kernel, dim3(1), dim3(64), 0, (hipStream_t)stream, a);
  hipError_t e = hipGetLastError(); if (e != hipSuccess) { fprintf(stderr, "[sdma] wait_kernel: %s\n", hipGetErrorString(e)); return -1; } return 0;
}
static int chk(hsa_status_t s, const char* w) { if (s != HSA_STATUS_SUCCESS) { const char* m; hsa_status_string(s, &m); fprintf(stderr, "[sdma] %s: %s\n", w, m); return -1; } return 0; }
static int hchk(hipError_t e, const char* w) { if (e != hipSuccess) { fprintf(stderr, "[sdma] %s: %s\n", w, hipGetErrorString(e)); return -1; } return 0; }
int sd_init() { return chk(hsa_init(), "hsa_init"); }
uint64_t sd_agent(void* p) {
  hsa_amd_pointer_info_t info; memset(&info, 0, sizeof(info)); info.size = sizeof(info);
  if (chk(hsa_amd_pointer_info(p, &info, nullptr, nullptr, nullptr), "pointer_info")) return 0;
  return info.agentOwner.handle;
}
int sd_ptype(void* p) {
  hsa_amd_pointer_info_t info; memset(&info, 0, sizeof(info)); info.size = sizeof(info);
  hsa_amd_pointer_info(p, &info, nullptr, nullptr, nullptr); return (int)info.type;
}
uint32_t sd_engines(uint64_t dst, uint64_t src) { uint32_t m = 0; if (chk(hsa_amd_memory_copy_engine_status(hsa_agent_t{dst}, hsa_agent_t{src}, &m), "status")) return 0; return m; }
uint32_t sd_preferred(uint64_t dst, uint64_t src) { uint32_t m = 0; if (chk(hsa_amd_memory_get_preferred_copy_engine(hsa_agent_t{dst}, hsa_agent_t{src}, &m), "pref")) return 0; return m; }
uint64_t sd_signal() { hsa_signal_t s; if (chk(hsa_signal_create(0, 0, nullptr, &s), "signal")) return 0; return s.handle; }
void sd_signal_set(uint64_t s, int64_t v) { hsa_signal_store_screlease(hsa_signal_t{s}, v); }
int64_t sd_signal_load(uint64_t s) { return hsa_signal_load_scacquire(hsa_signal_t{s}); }
// address of the signal's value (amd_signal_t: int64 kind; int64 value; ...)
void* sd_signal_addr(uint64_t s) { return (void*)(s + 8); }
int64_t sd_wait(uint64_t s) { return hsa_signal_wait_scacquire(hsa_signal_t{s}, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX, HSA_WAIT_STATE_ACTIVE); }
int sd_copy(void* dst, uint64_t dsta, const void* src, uint64_t srca, size_t n, uint64_t sig, uint32_t engine, int ndep, const uint64_t* deps) {
  if (engine == 0)
    return chk(hsa_amd_memory_async_copy(dst, hsa_agent_t{dsta}, src, hsa_agent_t{srca}, n, ndep, (const hsa_signal_t*)deps, hsa_signal_t{sig}), "copy");
  return chk(hsa_amd_memory_async_copy_on_engine(dst, hsa_agent_t{dsta}, src, hsa_agent_t{srca}, n, ndep, (const hsa_signal_t*)deps,
             hsa_signal_t{sig}, (hsa_amd_sdma_engine_id_t)engine, true), "copy_on_engine");
}
// HIP side
int sd_ipc_get(void* p, void* handle64) { return hchk(hipIpcGetMemHandle((hipIpcMemHandle_t*)handle64, p), "ipc_get"); }
void* sd_ipc_open(const void* handle64) { void* p = nullptr; hipIpcMemHandle_t h; memcpy(&h, handle64, sizeof(h));
  if (hchk(hipIpcOpenMemHandle(&p, h, hipIpcMemLazyEnablePeerAccess), "ipc_open")) return nullptr; return p; }
int sd_ipc_close(void* p) { return hchk(hipIpcCloseMemHandle(p), "ipc_close"); }
int sd_handle_size() { return (int)sizeof(hipIpcMemHandle_t); }
int sd_stream_write(void* stream, void* addr, int64_t v) { return hchk(hipStreamWriteValue64((hipStream_t)stream, addr, (uint64_t)v, 0), "write_value"); }
// wait until *addr >= v  (flags are monotonic sequence numbers)
int sd_stream_wait_ge(void* stream, void* addr, int64_t v) { return hchk(hipStreamWaitValue64((hipStream_t)stream, addr, (uint64_t)v, hipStreamWaitValueGte, ~0ull), "wait_value"); }
int sd_stream_wait_eq0(void* stream, void* addr) { return hchk(hipStreamWaitValue64((hipStream_t)stream, addr, 0ull, hipStreamWaitValueEq, ~0ull), "wait_eq0"); }
int sd_peer(int a, int b) { hipSetDevice(a); hipError_t e = hipDeviceEnablePeerAccess(b, 0); return (e == hipSuccess || e == hipErrorPeerAccessAlreadyEnabled) ? 0 : (int)e; }
}
