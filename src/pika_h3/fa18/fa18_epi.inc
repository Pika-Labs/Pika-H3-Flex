# SPDX-License-Identifier: Apache-2.0
# ---------------- epilogue ----------------
lab('.Lepi')
nops(19)
e('s_waitcnt vmcnt(0) lgkmcnt(0)')
if ROWSUM in ('smfmac', 'sm16'):   # lt = C[4hh][q] + C[4hh+1][q]; write l such that l + swap(l) = lt
    e(f'v_add_f32_e32 {vr(V_L)}, {vr(V_LACC)}, {vr(V_LACC+1)}')
    e(f'v_mul_f32_e32 {vr(V_L)}, 0.5, {vr(V_L)}')
e(f'v_mov_b32_e32 {vr(V_T)}, {vr(V_L)}')
nops(2)
e(f'v_permlane32_swap_b32_e32 {vr(V_T)}, {vr(V_L)}')
nops(2)
e(f'v_add_f32_e32 {vr(V_L)}, {vr(V_T)}, {vr(V_L)}')                   # lt (same in both halves)
e(f'v_rcp_f32_e32 {vr(V_MC)}, {vr(V_L)}')
e(f'v_cmp_gt_u32_e32 vcc, {sr(S_N)}, {vr(V_ROW)}')
nops(4)
e('s_and_saveexec_b64 s[76:77], vcc')
e('s_cbranch_execz .Lend')
# out offset = row*so_s*2 + hh*8
e(f'v_mul_lo_u32 {vr(V_T)}, {vr(V_ROW)}, {sr(S_SOS)}')
e(f'v_lshlrev_b32_e32 {vr(V_T)}, 1, {vr(V_T)}')
e(f'v_add_u32_e32 {vr(V_T)}, {vr(V_T)}, {vr(V_HH8)}')
nops(1)
for i in range(64): e(f'v_mul_f32_e32 {vr(V_O+i)}, {vr(V_MC)}, {vr(V_O+i)}')
for n in range(4):
    for g in range(4):
        b = V_O + 16 * n + 4 * g
        e(f'v_cvt_pk_bf16_f32 {vr(V_S[0]+8*n+2*g)}, {vr(b)}, {vr(b+1)}')
        e(f'v_cvt_pk_bf16_f32 {vr(V_S[0]+8*n+2*g+1)}, {vr(b+2)}, {vr(b+3)}')
for n in range(4):
    for g in range(4):
        e(f'global_store_dwordx2 {vr(V_T)}, {vr(V_S[0]+8*n+2*g, 2)}, {sr(S_OB, 2)} offset:{(n*32+g*8)*2}')
# lse (natural log): m*scale + ln(lt); lanes hh == 0 only
e(f's_cmp_eq_u64 {sr(S_LSE, 2)}, 0')
e('s_cbranch_scc1 .Lend')
e(f'v_cmp_eq_u32_e32 vcc, 0, {vr(V_HH8)}')
nops(4)
e('s_and_b64 exec, exec, vcc')
e('s_cbranch_execz .Lend')
e(f'v_log_f32_e32 {vr(V_X[0])}, {vr(V_L)}')
e(f'v_mul_f32_e32 {vr(V_X[1])}, {sr(S_SCALE)}, {vr(V_M)}')
nops(1)
e(f'v_fmac_f32_e32 {vr(V_X[1])}, 0x3f317218, {vr(V_X[0])}')           # ln 2
e(f's_mul_i32 {sr(S_TMP)}, {sr(S_N)}, s3')
e(f'v_add_u32_e32 {vr(V_T)}, {sr(S_TMP)}, {vr(V_ROW)}')
e(f'v_lshlrev_b32_e32 {vr(V_T)}, 2, {vr(V_T)}')
e(f'global_store_dword {vr(V_T)}, {vr(V_X[1])}, {sr(S_LSE, 2)}')
lab('.Lend')
e('s_endpgm')
L.append(f'\t.size {NAME}, .-{NAME}')
# ---------------- descriptor + metadata ----------------
L.append('\t.rodata')
L.append('\t.p2align 6')
L.append(f'\t.amdhsa_kernel {NAME}')
for kv in [('group_segment_fixed_size', 131072), ('private_segment_fixed_size', 0), ('kernarg_size', 120),
           ('user_sgpr_count', 2), ('user_sgpr_kernarg_segment_ptr', 1), ('system_sgpr_workgroup_id_x', 1),
           ('system_sgpr_workgroup_id_y', 1), ('system_vgpr_workitem_id', 0), ('next_free_vgpr', NVGPR),
           ('next_free_sgpr', NSGPR), ('accum_offset', NVGPR), ('reserve_vcc', 1), ('float_denorm_mode_32', 3),
           ('float_denorm_mode_16_64', 3), ('dx10_clamp', 1), ('ieee_mode', 1)]:
    L.append(f'\t\t.amdhsa_{kv[0]} {kv[1]}')
L.append('\t.end_amdhsa_kernel')
L.append(f"""\t.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           120
        .value_kind:     by_value
    .group_segment_fixed_size: 131072
    .kernarg_segment_align: 8
    .kernarg_segment_size: 120
    .max_flat_workgroup_size: 512
    .name:           {NAME}
    .private_segment_fixed_size: 0
    .sgpr_count:     {NSGPR}
    .sgpr_spill_count: 0
    .symbol:         {NAME}.kd
    .uniform_work_group_size: 1
    .vgpr_count:     {NVGPR}
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...
\t.end_amdgpu_metadata""")
open(sys.argv[1], 'w').write('\n'.join(L) + '\n')
print(f'{sys.argv[1]}: {len(L)} lines, LA={LA}')
