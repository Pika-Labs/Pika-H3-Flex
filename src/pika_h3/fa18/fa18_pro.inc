# SPDX-License-Identifier: Apache-2.0
# ================= kernel =================
L.append('\t.amdgcn_target "amdgcn-amd-amdhsa--gfx950"')
L.append('\t.amdhsa_code_object_version 6')
L.append('\t.text')
L.append(f'\t.globl {NAME}')
L.append('\t.p2align 8')
L.append(f'\t.type {NAME},@function')
lab(NAME)
e(f's_load_dwordx16 {sr(8, 16)}, s[0:1], 0x0')
e(f's_load_dwordx8 {sr(24, 8)}, s[0:1], 0x40')
e(f's_load_dwordx4 {sr(32, 4)}, s[0:1], 0x60')
e(f's_load_dword {sr(36)}, s[0:1], 0x70')
# lane quantities
e(f'v_and_b32_e32 {vr(V_TID)}, 0x3ff, {vr(V_TID)}')
e(f'v_lshrrev_b32_e32 {vr(V_X[0])}, 6, {vr(V_TID)}')
nops(4)
e(f'v_readfirstlane_b32 {sr(S_WAVE)}, {vr(V_X[0])}')
nops(4)
e(f'v_and_b32_e32 {vr(V_TID)}, 63, {vr(V_TID)}')                     # v0 = lane
e(f'v_lshlrev_b32_e32 {vr(V_L0)}, 4, {vr(V_TID)}')                    # lane*16
e(f'v_add_u32_e32 {vr(V_L1)}, 0x10000, {vr(V_L0)}')
e(f'v_lshrrev_b32_e32 {vr(V_HH8)}, 5, {vr(V_TID)}')
e(f'v_lshlrev_b32_e32 {vr(V_HH8)}, 3, {vr(V_HH8)}')                   # hh*8
e(f's_lshl_b32 {sr(S_WB)}, {sr(S_WAVE)}, 10')                          # wave*1024
e(f's_mov_b32 {sr(S_ZERO)}, 0')
e('s_waitcnt lgkmcnt(0)')
# nt, ntm1, masktile
e(f's_add_u32 {sr(S_TMP)}, {sr(S_N)}, 63')
e(f's_lshr_b32 {sr(S_TMP)}, {sr(S_TMP)}, 6')
e(f's_sub_u32 {sr(S_NTM1)}, {sr(S_TMP)}, 1')
e(f's_and_b32 {sr(S_TMP)}, {sr(S_N)}, 63')
e(f's_cmp_eq_u32 {sr(S_TMP)}, 0')
e(f's_cselect_b32 {sr(S_MASKT)}, -1, {sr(S_NTM1)}')
e(f's_mul_i32 {sr(S_KS64)}, {sr(S_SKS)}, 128')                         # 64 rows * sk_s * 2 bytes
# K resource: base = k + head*sk_h*2, num_records = ((N-1)*sk_s + 128)*2
def base_add(dst, src, stride_lo, idx_s):
    e(f's_mul_i32 {sr(S_TMP)}, {sr(stride_lo)}, {sr(idx_s)}')
    e(f's_mul_hi_u32 {sr(S_TMP2)}, {sr(stride_lo)}, {sr(idx_s)}')
    # 64-bit (tmp2:tmp) <<= 1, then add
    e(f's_lshl_b32 {sr(S_TMP2)}, {sr(S_TMP2)}, 1')
    e(f's_lshr_b32 {sr(S_TMP+1)}, {sr(S_TMP)}, 31')
    e(f's_or_b32 {sr(S_TMP2)}, {sr(S_TMP2)}, {sr(S_TMP+1)}')
    e(f's_lshl_b32 {sr(S_TMP)}, {sr(S_TMP)}, 1')
    e(f's_add_u32 {sr(dst)}, {sr(src)}, {sr(S_TMP)}')
    e(f's_addc_u32 {sr(dst+1)}, {sr(src+1)}, {sr(S_TMP2)}')
base_add(S_RK, S_K, S_SKH, 3)
e(f's_and_b32 {sr(S_RK+1)}, {sr(S_RK+1)}, 0xffff')
e(f's_sub_u32 {sr(S_TMP)}, {sr(S_N)}, 1')
e(f's_mul_i32 {sr(S_TMP)}, {sr(S_TMP)}, {sr(S_SKS)}')
e(f's_add_u32 {sr(S_TMP)}, {sr(S_TMP)}, 128')
e(f's_lshl_b32 {sr(S_RK+2)}, {sr(S_TMP)}, 1')
e(f's_mov_b32 {sr(S_RK+3)}, 0x00020000')
base_add(S_RV, S_VT, S_SVTH, 3)
e(f's_and_b32 {sr(S_RV+1)}, {sr(S_RV+1)}, 0xffff')
e(f's_mul_i32 {sr(S_TMP)}, {sr(S_SVTD)}, 256')
e(f's_mov_b32 {sr(S_RV+2)}, {sr(S_TMP)}')
e(f's_mov_b32 {sr(S_RV+3)}, 0x00020000')
base_add(S_QB, S_Q, S_SQH, 3)
base_add(S_OB, S_O, S_SOH, 3)
