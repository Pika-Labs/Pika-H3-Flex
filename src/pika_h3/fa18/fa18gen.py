# SPDX-License-Identifier: Apache-2.0
"""fa18: fa17 with 128-key tiles (half the barrier transitions per flop). hd 128, bf16 in/out, fp32 scores/softmax/accum.
Same layout/numerics as fa17 (S^T = K Q^T, queries along lanes, 32 rows/wave, permuted K rows, V^T [h, D, Np], LDS-DMA,
b128 LDS reads, exact lazy rescale, row sum = 4 x v_smfmac_f32_16x16x64_bf16 on bf16 P).
Schedule: 2 wave groups, 2 barriers per tile; MFMA phase = 4 smfmac + QK(t+1) (32 MFMA, 4 accumulators) + PV(t) (32 MFMA);
softmax phase = 64 scores per lane.  Single S buffer (QK(t+1) overwrites S after softmax(t) moved P to its own 32 regs).
LDS: K ring of 3 x 32 KB, V ring of 2 x 32 KB (160 KB).  Loads in MFMA(t): V(t+1) first, then K(t+3).
Barrier waits: vmcnt(4) after an MFMA phase (this phase's K loads may stay), vmcnt(0) after a softmax phase.
usage: python fa18gen.py out.s [LA=6] ..."""
import sys
args = dict(a.split('=') for a in sys.argv[2:])
LA = int(args.get('LA', 6)); PRIO = int(args.get('PRIO', 1)); NAME = args.get('NAME', 'fa17_fwd')
SMNOP2 = int(args.get('SMNOP2', 4)); TAU = float(args.get('TAU', 0))
VPOS = [int(x) for x in args.get('VPOS', '2,6,10,14').split(',')]
KPOS = [int(x) for x in args.get('KPOS', '34,38,42,46').split(',')]
ROWSUM = 'sm16'; NF = 6
L = []; OOLL = []; CUR = [L]
def e(s): CUR[0].append('\t' + s)
def lab(s): CUR[0].append(s + ':')
def vr(b, n=1): return f'v{b}' if n == 1 else f'v[{b}:{b+n-1}]'
def sr(b, n=1): return f's{b}' if n == 1 else f's[{b}:{b+n-1}]'
def nops(n):
    while n > 0:
        k = min(n, 8); e(f's_nop {k-1}'); n -= k
# ---------------- registers ----------------
V_TID, V_L0, V_L1, V_M, V_L, V_MX, V_T, V_L2, V_SIDX, V_MC = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
V_LDOK, V_LDOV = 10, 11
V_LDTK = [12, 13, 14, 15]; V_LDTV = 16
V_NVL, V_NINF, V_ROW, V_HH8 = 18, 19, 20, 21
V_X = [22, 23]
V_F = 24                       # 6 frags v[24:47]
V_ONES, V_LACC = 48, 52
V_O, V_Q, V_S0, V_P = 64, 128, 160, 224
V_S = [V_S0]
NVGPR = 256
S_Q, S_K, S_VT, S_O, S_LSE = 8, 10, 12, 14, 16
S_SQS, S_SQH, S_SKS = 18, 20, 22
S_SKH, S_SVTH, S_SVTD, S_SOS = 24, 26, 28, 30
S_SOH, S_N, S_C = 32, 34, 35
S_SCALE = 36
S_RK, S_RV = 40, 44
S_WAVE, S_WB, S_T, S_NTM1, S_MASKT, S_KS64 = 48, 49, 50, 51, 52, 53
S_TL, S_KOFF, S_VOFF, S_TMP, S_TMP2 = 54, 55, 56, 57, 59
S_QB, S_OB = 60, 62
S_MK = [64, 66, 68, 70]
S_RS32, S_KO = 78, [79, 80, 81]
S_VS128, S_VR32 = 82, 83
S_NQ = 37                       # query rows (kernarg 116); S_N = key rows
S_ZERO = 75
NSGPR = 96
KB = 32768
def kbuf(b): return b * KB
def vbuf(c): return (3 + c) * KB
def lds_addr(byteoff):
    base = [V_L0, V_L1, V_L2][byteoff >> 16]
    return vr(base), byteoff & 0xffff
def Fr(r): return vr(V_F + 4 * (r % NF), 4)
def kfrag(ii): return (ii & 3) * 8 + (ii >> 2)        # QK MFMA ii -> K frag (j = ii&3, d = ii>>2)
def read(r, u):
    """read r of the MFMA phase of slot u: r < 32 -> K(t+1) frag (buffer (u+1)%3), else V(t) frag r-32 (buffer u%2)"""
    off = kbuf((u + 1) % 3) + kfrag(r) * 1024 if r < 32 else vbuf(u % 2) + (r - 32) * 1024
    v, o = lds_addr(off); e(f'ds_read_b128 {Fr(r)}, {v} offset:{o}')
def Sr(i): return V_S0 + i

LDT0 = int(args.get('LDT0', 0))   # timing only: every tile load reads tile 0 (cache-resident K/V)
def loadsetup(tsym):
    # V tile min(t+1, nt-1), K tile min(t+3, nt-1); per-tile load addresses
    e(f's_add_u32 {sr(S_TL)}, {sr(tsym)}, 1'); e(f's_min_u32 {sr(S_TL)}, {sr(S_TL)}, {sr(S_NTM1)}')
    if LDT0: e(f's_mov_b32 {sr(S_TL)}, 0')
    e(f's_lshl_b32 {sr(S_VOFF)}, {sr(S_TL)}, 8')                     # 128 keys * 2 bytes
    e(f's_add_u32 {sr(S_TL)}, {sr(tsym)}, 3'); e(f's_min_u32 {sr(S_TL)}, {sr(S_TL)}, {sr(S_NTM1)}')
    if LDT0: e(f's_mov_b32 {sr(S_TL)}, 0')
    e(f's_mul_i32 {sr(S_KOFF)}, {sr(S_TL)}, {sr(S_KS64)}')            # here: 128 rows * sk_s * 2
    e(f'v_add_u32_e32 {vr(V_LDTV)}, {sr(S_VOFF)}, {vr(V_LDOV)}')
    e(f'v_add_u32_e32 {vr(V_LDTK[0])}, {sr(S_KOFF)}, {vr(V_LDOK)}')
    for r in range(1, 4):
        e(f's_add_u32 {sr(S_KO[r-1])}, {sr(S_KOFF)}, {sr(S_RS32)}' if r == 1 else f's_add_u32 {sr(S_KO[r-1])}, {sr(S_KO[r-2])}, {sr(S_RS32)}')
        e(f'v_add_u32_e32 {vr(V_LDTK[r])}, {sr(S_KO[r-1])}, {vr(V_LDOK)}')

def ld_v(r, vb_c):
    e(f's_add_u32 m0, {sr(S_WB)}, {vbuf(vb_c) + r * 8192 - r * 64}')   # inst offset also offsets the LDS address
    e(f'buffer_load_dwordx4 {vr(V_LDTV)}, {sr(S_RV, 4)}, 0 offen{f" offset:{r*64}" if r else ""} lds')
def ld_k(r, kb_b):
    e(f's_add_u32 m0, {sr(S_WB)}, {kbuf(kb_b) + r * 8192}')
    e(f'buffer_load_dwordx4 {vr(V_LDTK[r])}, {sr(S_RK, 4)}, 0 offen lds')

SPLITLD = int(args.get('SPLITLD', 0))
VTR = int(args.get('VTR', 0))   # V read from the strided [N, h, D] view (vt arg = v, svt_h = head stride, svt_d = row stride);
                               # LDS V tile byte(key, d) = (key>>2)*1024 + (d>>3)*64 + (key&3)*16 + (d&7)*2; PV frags by 2 x ds_read_b64_tr_b16
V_VB = [17, 56]                 # per-lane tr-read bases for V buffer 0 / 1
S_RV2 = 84                      # V resource shifted by 16 rows (odd load index)
if VTR: assert SPLITLD
SOFT = args.get('SOFT', 'full')
LPOS = [int(x) for x in args.get('LPOS', '4,12,20,28,36,44,52,60').split(',')]
def loadsetup_split(tsym, grpB):
    if grpB and VTR:   # V rows of tile min(t+1, nt-1): 4 VGPRs (32-row steps), odd loads via the +16-row resource
        e(f's_add_u32 {sr(S_TL)}, {sr(tsym)}, 1'); e(f's_min_u32 {sr(S_TL)}, {sr(S_TL)}, {sr(S_NTM1)}')
        e(f's_mul_i32 {sr(S_KOFF)}, {sr(S_TL)}, {sr(S_VS128)}')
        e(f'v_add_u32_e32 {vr(V_LDTK[0])}, {sr(S_KOFF)}, {vr(V_LDOV)}')
        for r in range(1, 4):
            e(f's_add_u32 {sr(S_KO[r-1])}, {sr(S_KOFF)}, {sr(S_VR32)}' if r == 1 else f's_add_u32 {sr(S_KO[r-1])}, {sr(S_KO[r-2])}, {sr(S_VR32)}')
            e(f'v_add_u32_e32 {vr(V_LDTK[r])}, {sr(S_KO[r-1])}, {vr(V_LDOV)}')
    elif grpB:   # V tile min(t+1, nt-1): one VGPR, chunk r at immediate offset r*32
        e(f's_add_u32 {sr(S_TL)}, {sr(tsym)}, 1'); e(f's_min_u32 {sr(S_TL)}, {sr(S_TL)}, {sr(S_NTM1)}')
        if LDT0: e(f's_mov_b32 {sr(S_TL)}, 0')
        e(f's_lshl_b32 {sr(S_VOFF)}, {sr(S_TL)}, 8')
        e(f'v_add_u32_e32 {vr(V_LDTV)}, {sr(S_VOFF)}, {vr(V_LDOV)}')
    else:      # K tile min(t+3, nt-1): 4 VGPRs (row block r>>1), odd r at immediate offset 128
        e(f's_add_u32 {sr(S_TL)}, {sr(tsym)}, 3'); e(f's_min_u32 {sr(S_TL)}, {sr(S_TL)}, {sr(S_NTM1)}')
        if LDT0: e(f's_mov_b32 {sr(S_TL)}, 0')
        e(f's_mul_i32 {sr(S_KOFF)}, {sr(S_TL)}, {sr(S_KS64)}')
        e(f'v_add_u32_e32 {vr(V_LDTK[0])}, {sr(S_KOFF)}, {vr(V_LDOK)}')
        for r in range(1, 4):
            e(f's_add_u32 {sr(S_KO[r-1])}, {sr(S_KOFF)}, {sr(S_RS32)}' if r == 1 else f's_add_u32 {sr(S_KO[r-1])}, {sr(S_KO[r-2])}, {sr(S_RS32)}')
            e(f'v_add_u32_e32 {vr(V_LDTK[r])}, {sr(S_KO[r-1])}, {vr(V_LDOK)}')
def ld_split(r, grpB, buf_off):
    if grpB and VTR:   # key group G = 4r + (wave&3): 1 KB at buffer + r*4096 + (wave&3)*1024
        e(f's_add_u32 m0, {sr(S_WB)}, {buf_off + r * 4096}')
        e(f'buffer_load_dwordx4 {vr(V_LDTK[r >> 1])}, {sr(S_RV2 if r & 1 else S_RV, 4)}, 0 offen lds')
    elif grpB:   # V chunk r: frag kc = r, n = wave&3 -> LDS buffer + r*4096 + (wave&3)*1024
        e(f's_add_u32 m0, {sr(S_WB)}, {buf_off + r * 4096 - r * 32}')
        e(f'buffer_load_dwordx4 {vr(V_LDTV)}, {sr(S_RV, 4)}, 0 offen{f" offset:{r*32}" if r else ""} lds')
    else:      # K chunk r: frag j = r>>1, d = (r&1)*4 + wave -> LDS buffer + r*4096 + wave*1024
        e(f's_add_u32 m0, {sr(S_WB)}, {buf_off + r * 4096 - (r & 1) * 128}')
        e(f'buffer_load_dwordx4 {vr(V_LDTK[r >> 1])}, {sr(S_RK, 4)}, 0 offen{" offset:128" if r & 1 else ""} lds')

def softmax(u, tsym, pre, prefetch=True):
    uid = f'{len(L)}'
    if PRIO: e('s_setprio 0')
    if prefetch:
        for r in range(LA): read(r, u)
    pre()
    nops(SMNOP2)
    e(f's_cmp_eq_u32 {sr(tsym)}, {sr(S_MASKT)}')
    e(f's_cbranch_scc1 .Lmask_{uid}')
    CUR[0] = OOLL; lab(f'.Lmask_{uid}')
    e(f's_lshl_b32 {sr(S_TMP)}, {sr(tsym)}, 7')
    e(f's_sub_u32 {sr(S_TMP)}, {sr(S_N)}, {sr(S_TMP)}')
    e(f'v_sub_u32_e32 {vr(V_NVL)}, {sr(S_TMP)}, {vr(V_HH8)}')
    e(f'v_subrev_u32_e32 {vr(V_X[1])}, 64, {vr(V_NVL)}')          # nv - hh*8 - 64 (inline constants only go to 64)
    for g in range(0, 64, 4):
        for q in range(4):
            i = g + q; j, ii = i // 16, i % 16
            kidx = j * 32 + (ii >> 3) * 16 + (ii & 7)
            if kidx < 64: e(f'v_cmp_lt_i32_e64 {sr(S_MK[q], 2)}, {kidx}, {vr(V_NVL)}')
            else: e(f'v_cmp_lt_i32_e64 {sr(S_MK[q], 2)}, {kidx - 64}, {vr(V_X[1])}')
        nops(2)
        for q in range(4):
            i = g + q
            e(f'v_cndmask_b32_e64 {vr(Sr(i))}, {vr(V_NINF)}, {vr(Sr(i))}, {sr(S_MK[q], 2)}')
    e(f's_branch .Lnomask_{uid}'); CUR[0] = L
    lab(f'.Lnomask_{uid}')
    if SOFT == 'cvt':   # TIMING/ENERGY ONLY: P = bf16(S * 2^-6) (no max, fma, exp, rescale)
        for i in range(32): e(f'v_cvt_pk_bf16_f32 {vr(V_P+i)}, {vr(Sr(2*i))}, {vr(Sr(2*i+1))}')
        return
    if SOFT == 'none':
        return
    e(f'v_mov_b32_e32 {vr(V_MX)}, {vr(V_M)}')
    for i in range(32): e(f'v_max3_f32 {vr(V_MX)}, {vr(Sr(2*i))}, {vr(Sr(2*i+1))}, {vr(V_MX)}')
    e(f'v_mov_b32_e32 {vr(V_T)}, {vr(V_MX)}'); nops(2)
    e(f'v_permlane32_swap_b32_e32 {vr(V_T)}, {vr(V_MX)}'); nops(2)
    e(f'v_max_f32_e32 {vr(V_MX)}, {vr(V_T)}, {vr(V_MX)}')
    if TAU > 0:
        e(f'v_sub_f32_e32 {vr(V_T)}, {vr(V_MX)}, {vr(V_M)}'); e(f'v_mul_f32_e32 {vr(V_T)}, {sr(S_C)}, {vr(V_T)}')
        e(f'v_cmp_le_f32_e32 vcc, {TAU}, {vr(V_T)}'); nops(2)
        e(f'v_cndmask_b32_e32 {vr(V_MX)}, {vr(V_M)}, {vr(V_MX)}, vcc')
    e(f'v_cmp_lt_f32_e32 vcc, {vr(V_M)}, {vr(V_MX)}')
    e(f's_cbranch_vccnz .Lresc_{uid}')
    CUR[0] = OOLL; lab(f'.Lresc_{uid}')
    e(f'v_sub_f32_e32 {vr(V_T)}, {vr(V_M)}, {vr(V_MX)}')
    e(f'v_mul_f32_e32 {vr(V_T)}, {sr(S_C)}, {vr(V_T)}')
    e(f'v_exp_f32_e32 {vr(V_T)}, {vr(V_T)}'); nops(1)
    e(f'v_mul_f32_e32 {vr(V_LACC)}, {vr(V_T)}, {vr(V_LACC)}')
    e(f'v_mul_f32_e32 {vr(V_LACC+1)}, {vr(V_T)}, {vr(V_LACC+1)}')
    for i in range(64): e(f'v_mul_f32_e32 {vr(V_O+i)}, {vr(V_T)}, {vr(V_O+i)}')
    e(f's_branch .Lnoresc_{uid}'); CUR[0] = L
    lab(f'.Lnoresc_{uid}')
    e(f'v_mov_b32_e32 {vr(V_M)}, {vr(V_MX)}')
    e(f'v_mul_f32_e32 {vr(V_MC)}, {sr(S_C)}, {vr(V_MX)}')
    for i in range(64): e(f'v_fma_f32 {vr(Sr(i))}, {vr(Sr(i))}, {sr(S_C)}, -{vr(V_MC)}')
    for i in range(64): e(f'v_exp_f32_e32 {vr(Sr(i))}, {vr(Sr(i))}')
    for i in range(32): e(f'v_cvt_pk_bf16_f32 {vr(V_P+i)}, {vr(Sr(2*i))}, {vr(Sr(2*i+1))}')

def mfma_phase(u, pv_only, grpB=False):
    nm = 32 if pv_only else 64
    for h in range(4):
        e(f'v_smfmac_f32_16x16x64_bf16 {vr(V_LACC, 4)}, {vr(V_ONES, 4)}, {vr(V_P + 8 * h, 8)}, {vr(V_SIDX)}')
    base = 32 if pv_only else 0
    # read list: (frag index = consuming MFMA, emitter)
    R = []
    for i in range(base, 64):
        if i < 32 or not VTR: R.append(i)
        else: R.append(i); R.append(i)
    def emit_read(k):
        f = R[k]
        if f < 32 or not VTR: read(f, u); return
        h = 0 if k == 0 or R[k - 1] != f else 1
        ii = f - 32; kc, n = ii >> 2, ii & 3
        e(f'ds_read_b64_tr_b16 {vr(V_F + 4 * (f % NF) + 2 * h, 2)}, {vr(V_VB[u % 2])} offset:{kc * 4096 + h * 1024 + n * 256}')
    last = {}
    for k, f in enumerate(R): last[f] = k
    if pv_only:
        e('s_waitcnt lgkmcnt(0)')
        p = 0
    else:
        p = sum(1 for f in R if f < LA)     # the softmax phase prefetched the reads of frags < LA (K frags)
    if pv_only:
        while p < len(R) and R[p] < base + LA: emit_read(p); p += 1
    vi = ki = 0
    for i in range(base, 64):
        e(f's_waitcnt lgkmcnt({min(p - last[i] - 1, 15)})')
        if i < 32:
            j, d = i & 3, i >> 2
            c = '0' if d == 0 else vr(V_S0 + 16 * j, 16)
            e(f'v_mfma_f32_32x32x16_bf16 {vr(V_S0+16*j, 16)}, {Fr(i)}, {vr(V_Q+4*d, 4)}, {c}')
        else:
            ii = i - 32; kc, n = ii >> 2, ii & 3
            e(f'v_mfma_f32_32x32x16_bf16 {vr(V_O+16*n, 16)}, {Fr(i)}, {vr(V_P+4*kc, 4)}, {vr(V_O+16*n, 16)}')
        while p < len(R) and R[p] < i + 1 + LA: emit_read(p); p += 1
        if not pv_only and SPLITLD:
            if vi < 8 and i == LPOS[vi]: ld_split(vi, grpB, vbuf((u + 1) % 2) if grpB else kbuf(u % 3)); vi += 1
        elif not pv_only:
            if vi < 4 and i == VPOS[vi]: ld_v(vi, (u + 1) % 2); vi += 1
            if ki < 4 and i == KPOS[ki]: ld_k(ki, u % 3); ki += 1

def barrier(vm):
    e(f's_waitcnt vmcnt({vm})'); e('s_barrier')

exec(open(''+__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/fa18_pro.inc').read())
e(f's_load_dword {sr(S_NQ)}, s[0:1], 0x74'); e('s_waitcnt lgkmcnt(0)')
# 128-key tiles: nt, ntm1, mask tile, K stride per tile
e(f's_add_u32 {sr(S_TMP)}, {sr(S_N)}, 127'); e(f's_lshr_b32 {sr(S_TMP)}, {sr(S_TMP)}, 7')
e(f's_sub_u32 {sr(S_NTM1)}, {sr(S_TMP)}, 1')
e(f's_and_b32 {sr(S_TMP)}, {sr(S_N)}, 127'); e(f's_cmp_eq_u32 {sr(S_TMP)}, 0')
e(f's_cselect_b32 {sr(S_MASKT)}, -1, {sr(S_NTM1)}')
e(f's_mul_i32 {sr(S_KS64)}, {sr(S_SKS)}, 256')
e(f's_mul_i32 {sr(S_RS32)}, {sr(S_SKS)}, 64')
e(f'v_add_u32_e32 {vr(V_L2)}, 0x20000, {vr(V_L0)}')
# per-lane load offsets (as fa17): K: (pi32(l&31)*sk_s + wave*16 + hh*8)*2 ; V: ((n*32 + (l&31))*svt_d + (wave>>2)*16 + hh*8)*2
e(f'v_and_b32_e32 {vr(V_X[0])}, 31, {vr(V_TID)}')
e(f'v_and_b32_e32 {vr(V_X[1])}, 0x13, {vr(V_X[0])}')
e(f'v_and_b32_e32 {vr(V_T)}, 4, {vr(V_X[0])}')
e(f'v_lshl_or_b32 {vr(V_X[1])}, {vr(V_T)}, 1, {vr(V_X[1])}')
e(f'v_and_b32_e32 {vr(V_T)}, 8, {vr(V_X[0])}')
e(f'v_lshrrev_b32_e32 {vr(V_T)}, 1, {vr(V_T)}')
e(f'v_or_b32_e32 {vr(V_X[1])}, {vr(V_X[1])}, {vr(V_T)}')
e(f's_lshl_b32 {sr(S_TMP)}, {sr(S_WAVE)}, 4')
e(f'v_add_u32_e32 {vr(V_T)}, {sr(S_TMP)}, {vr(V_HH8)}')
e(f'v_mul_lo_u32 {vr(V_MX)}, {vr(V_X[1])}, {sr(S_SKS)}')
e(f'v_add_u32_e32 {vr(V_MX)}, {vr(V_MX)}, {vr(V_T)}')
e(f'v_lshlrev_b32_e32 {vr(V_LDOK)}, 1, {vr(V_MX)}')
e(f's_and_b32 {sr(S_TMP)}, {sr(S_WAVE)}, 3'); e(f's_lshl_b32 {sr(S_TMP)}, {sr(S_TMP)}, 5')
e(f'v_add_u32_e32 {vr(V_T)}, {sr(S_TMP)}, {vr(V_X[0])}')
e(f'v_mul_lo_u32 {vr(V_T)}, {vr(V_T)}, {sr(S_SVTD)}')
e(f's_lshr_b32 {sr(S_TMP)}, {sr(S_WAVE)}, 2'); e(f's_lshl_b32 {sr(S_TMP)}, {sr(S_TMP)}, 4')
e(f'v_add_u32_e32 {vr(V_T)}, {sr(S_TMP)}, {vr(V_T)}')
e(f'v_add_u32_e32 {vr(V_T)}, {vr(V_T)}, {vr(V_HH8)}')
e(f'v_lshlrev_b32_e32 {vr(V_LDOV)}, 1, {vr(V_T)}')
# prologue loads: K(0), K(1), K(2) -> K buffers 0..2, V(0) -> V buffer 0
for t in range(3):
    e(f's_mov_b32 {sr(S_TL)}, {t}'); e(f's_min_u32 {sr(S_TL)}, {sr(S_TL)}, {sr(S_NTM1)}')
    e(f's_mul_i32 {sr(S_KOFF)}, {sr(S_TL)}, {sr(S_KS64)}')
    e(f'v_add_u32_e32 {vr(V_LDTK[0])}, {sr(S_KOFF)}, {vr(V_LDOK)}')
    for r in range(1, 4):
        e(f's_add_u32 {sr(S_KO[r-1])}, {sr(S_KOFF)}, {sr(S_RS32)}' if r == 1 else f's_add_u32 {sr(S_KO[r-1])}, {sr(S_KO[r-2])}, {sr(S_RS32)}')
        e(f'v_add_u32_e32 {vr(V_LDTK[r])}, {sr(S_KO[r-1])}, {vr(V_LDOK)}')
    for r in range(4): ld_k(r, t)
    e('s_waitcnt vmcnt(0)')
if VTR:
    # V resource over the strided view (vt arg = v, svt_h = head stride, svt_d = row stride); num_records like K; +16-row copy
    e(f's_mul_i32 {sr(S_TMP)}, {sr(S_SVTH)}, s3'); e(f's_mul_hi_u32 {sr(S_TMP2)}, {sr(S_SVTH)}, s3')
    e(f's_lshl_b32 {sr(S_TMP2)}, {sr(S_TMP2)}, 1'); e(f's_lshr_b32 {sr(S_TMP+1)}, {sr(S_TMP)}, 31')
    e(f's_or_b32 {sr(S_TMP2)}, {sr(S_TMP2)}, {sr(S_TMP+1)}'); e(f's_lshl_b32 {sr(S_TMP)}, {sr(S_TMP)}, 1')
    e(f's_add_u32 {sr(S_RV)}, {sr(S_VT)}, {sr(S_TMP)}'); e(f's_addc_u32 {sr(S_RV+1)}, {sr(S_VT+1)}, {sr(S_TMP2)}')
    e(f's_and_b32 {sr(S_RV+1)}, {sr(S_RV+1)}, 0xffff')
    e(f's_sub_u32 {sr(S_TMP)}, {sr(S_N)}, 1'); e(f's_mul_i32 {sr(S_TMP)}, {sr(S_TMP)}, {sr(S_SVTD)}')
    e(f's_add_u32 {sr(S_TMP)}, {sr(S_TMP)}, 128'); e(f's_lshl_b32 {sr(S_RV+2)}, {sr(S_TMP)}, 1')
    e(f's_mov_b32 {sr(S_RV+3)}, 0x00020000')
    e(f's_mul_i32 {sr(S_VS128)}, {sr(S_SVTD)}, 256'); e(f's_mul_i32 {sr(S_VR32)}, {sr(S_SVTD)}, 64')
    e(f's_mul_i32 {sr(S_TMP)}, {sr(S_SVTD)}, 32')                       # 16 rows in bytes
    e(f's_add_u32 {sr(S_RV2)}, {sr(S_RV)}, {sr(S_TMP)}'); e(f's_addc_u32 {sr(S_RV2+1)}, {sr(S_RV+1)}, 0')
    e(f's_sub_u32 {sr(S_RV2+2)}, {sr(S_RV+2)}, {sr(S_TMP)}'); e(f's_mov_b32 {sr(S_RV2+3)}, 0x00020000')
    # per-lane V load offset, all 8 waves (prologue): key 4w + (i&3), dims (i>>2)*8
    e(f'v_and_b32_e32 {vr(V_T)}, 3, {vr(V_TID)}'); e(f's_lshl_b32 {sr(S_TMP)}, {sr(S_WAVE)}, 2')
    e(f'v_add_u32_e32 {vr(V_T)}, {sr(S_TMP)}, {vr(V_T)}'); e(f'v_mul_lo_u32 {vr(V_T)}, {vr(V_T)}, {sr(S_SVTD)}')
    e(f'v_lshrrev_b32_e32 {vr(V_MX)}, 2, {vr(V_TID)}'); e(f'v_lshl_add_u32 {vr(V_T)}, {vr(V_MX)}, 3, {vr(V_T)}')
    e(f'v_lshlrev_b32_e32 {vr(V_LDOV)}, 1, {vr(V_T)}')
    # V(0): 32 key groups, G = 8r + w (r = 0..3): rows 32r + 4w; LDS 1 KB at vbuf(0) + G*1024
    for r in range(4):
        e(f's_mul_i32 {sr(S_TMP)}, {sr(S_SVTD)}, {64 * r}')
        e(f'v_add_u32_e32 {vr(V_LDTV)}, {sr(S_TMP)}, {vr(V_LDOV)}')
        e(f's_add_u32 m0, {sr(S_WB)}, {vbuf(0) + r * 8192}')
        e(f'buffer_load_dwordx4 {vr(V_LDTV)}, {sr(S_RV, 4)}, 0 offen lds')
        e('s_waitcnt vmcnt(0)')
    # loop mapping for B waves: key 4(w&3) + (i&3): subtract 16*(w>>2) rows
    e(f's_lshr_b32 {sr(S_TMP)}, {sr(S_WAVE)}, 2'); e(f's_mul_i32 {sr(S_TMP)}, {sr(S_TMP)}, {sr(S_SVTD)}')
    e(f's_lshl_b32 {sr(S_TMP)}, {sr(S_TMP)}, 5'); e(f'v_subrev_u32_e32 {vr(V_LDOV)}, {sr(S_TMP)}, {vr(V_LDOV)}')
    # tr-read bases: g = l>>4, j = l&15, s = j>>2, q = j&3: (g>>1)*2048 + (g&1)*128 + (q>>1)*64 + s*16 + (q&1)*8
    e(f'v_bfe_u32 {vr(V_T)}, {vr(V_TID)}, 5, 1'); e(f'v_lshlrev_b32_e32 {vr(V_T)}, 11, {vr(V_T)}')
    e(f'v_bfe_u32 {vr(V_MX)}, {vr(V_TID)}, 4, 1'); e(f'v_lshl_add_u32 {vr(V_T)}, {vr(V_MX)}, 7, {vr(V_T)}')
    e(f'v_bfe_u32 {vr(V_MX)}, {vr(V_TID)}, 1, 1'); e(f'v_lshl_add_u32 {vr(V_T)}, {vr(V_MX)}, 6, {vr(V_T)}')
    e(f'v_bfe_u32 {vr(V_MX)}, {vr(V_TID)}, 2, 2'); e(f'v_lshl_add_u32 {vr(V_T)}, {vr(V_MX)}, 4, {vr(V_T)}')
    e(f'v_and_b32_e32 {vr(V_MX)}, 1, {vr(V_TID)}'); e(f'v_lshl_add_u32 {vr(V_T)}, {vr(V_MX)}, 3, {vr(V_T)}')
    e(f'v_add_u32_e32 {vr(V_VB[0])}, {vbuf(0)}, {vr(V_T)}'); e(f'v_add_u32_e32 {vr(V_VB[1])}, {vbuf(1)}, {vr(V_T)}')
else:
    e(f'v_mov_b32_e32 {vr(V_LDTV)}, {vr(V_LDOV)}')
    for r in range(4): ld_v(r, 0)
if int(args.get('COAL', 0)):   # TIMING/ENERGY ONLY: row-contiguous per-lane load addresses (wrong layout, real finite data)
    # K: lane l -> row (l >> 4) (+ wave*4), chunk (l & 15); V^T: row d = (l >> 4) + wave*4, chunk (l & 15)
    e(f'v_lshrrev_b32_e32 {vr(V_T)}, 4, {vr(V_TID)}'); e(f's_lshl_b32 {sr(S_TMP)}, {sr(S_WAVE)}, 2')
    e(f'v_add_u32_e32 {vr(V_T)}, {sr(S_TMP)}, {vr(V_T)}')
    e(f'v_mul_lo_u32 {vr(V_MX)}, {vr(V_T)}, {sr(S_SKS)}'); e(f'v_lshlrev_b32_e32 {vr(V_MX)}, 1, {vr(V_MX)}')
    e(f'v_and_b32_e32 {vr(V_X[1])}, 15, {vr(V_TID)}'); e(f'v_lshlrev_b32_e32 {vr(V_X[1])}, 4, {vr(V_X[1])}')
    e(f'v_add_u32_e32 {vr(V_LDOK)}, {vr(V_MX)}, {vr(V_X[1])}')
    e(f'v_mul_lo_u32 {vr(V_MX)}, {vr(V_T)}, {sr(S_SVTD)}'); e(f'v_lshlrev_b32_e32 {vr(V_MX)}, 1, {vr(V_MX)}')
    e(f'v_add_u32_e32 {vr(V_LDOV)}, {vr(V_MX)}, {vr(V_X[1])}')
if SPLITLD and not VTR:   # loop mapping: B waves load all of V (no (wave>>2)*16 term), m0 base (wave&3)*1024
    e(f's_lshr_b32 {sr(S_TMP)}, {sr(S_WAVE)}, 2'); e(f's_lshl_b32 {sr(S_TMP)}, {sr(S_TMP)}, 5')
    e(f'v_subrev_u32_e32 {vr(V_LDOV)}, {sr(S_TMP)}, {vr(V_LDOV)}')
    e(f's_and_b32 {sr(S_TMP)}, {sr(S_WAVE)}, 3'); e('s_waitcnt vmcnt(0)'); e(f's_lshl_b32 {sr(S_WB)}, {sr(S_TMP)}, 10')
if VTR:
    e(f's_and_b32 {sr(S_TMP)}, {sr(S_WAVE)}, 3'); e('s_waitcnt vmcnt(0)'); e(f's_lshl_b32 {sr(S_WB)}, {sr(S_TMP)}, 10')
# Q
e(f's_lshl_b32 {sr(S_TMP)}, s2, 8'); e(f's_lshl_b32 {sr(S_TMP2)}, {sr(S_WAVE)}, 5')
e(f's_add_u32 {sr(S_TMP)}, {sr(S_TMP)}, {sr(S_TMP2)}')
e(f'v_add_u32_e32 {vr(V_ROW)}, {sr(S_TMP)}, {vr(V_X[0])}')
e(f's_sub_u32 {sr(S_TMP)}, {sr(S_NQ)}, 1')
e(f'v_min_u32_e32 {vr(V_T)}, {sr(S_TMP)}, {vr(V_ROW)}')
e(f'v_mul_lo_u32 {vr(V_T)}, {vr(V_T)}, {sr(S_SQS)}')
e(f'v_lshlrev_b32_e32 {vr(V_T)}, 1, {vr(V_T)}')
e(f'v_lshl_add_u32 {vr(V_T)}, {vr(V_HH8)}, 1, {vr(V_T)}')
for d in range(8): e(f'global_load_dwordx4 {vr(V_Q+4*d, 4)}, {vr(V_T)}, {sr(S_QB, 2)} offset:{d*32}')
e(f'v_mov_b32_e32 {vr(V_M)}, 0xf149f2ca'); e(f'v_mov_b32_e32 {vr(V_L)}, 0')
e(f'v_mov_b32_e32 {vr(V_NINF)}, 0xff800000')
for i in range(4): e(f'v_mov_b32_e32 {vr(V_ONES+i)}, 0x3f803f80')
for i in range(4): e(f'v_mov_b32_e32 {vr(V_LACC+i)}, 0')
e(f'v_bfe_u32 {vr(V_T)}, {vr(V_TID)}, 2, 1'); e(f'v_cmp_eq_u32_e32 vcc, 0, {vr(V_T)}'); nops(4)
e(f'v_cndmask_b32_e64 {vr(V_ONES+0)}, 0, {vr(V_ONES+0)}, vcc'); e(f'v_cndmask_b32_e64 {vr(V_ONES+1)}, 0, {vr(V_ONES+1)}, vcc')
e(f'v_cndmask_b32_e64 {vr(V_ONES+2)}, {vr(V_ONES+2)}, 0, vcc'); e(f'v_cndmask_b32_e64 {vr(V_ONES+3)}, {vr(V_ONES+3)}, 0, vcc')
e(f'v_and_b32_e32 {vr(V_T)}, 1, {vr(V_TID)}')
e(f'v_mov_b32_e32 {vr(V_SIDX)}, 0x4444'); e(f'v_mov_b32_e32 {vr(V_X[1])}, 0xeeee')
e(f'v_cmp_ne_u32_e32 vcc, 0, {vr(V_T)}'); nops(4)
e(f'v_cndmask_b32_e32 {vr(V_SIDX)}, {vr(V_SIDX)}, {vr(V_X[1])}, vcc')
for i in range(64): e(f'v_mov_b32_e32 {vr(V_O+i)}, 0')
e('s_waitcnt vmcnt(0)'); e('s_barrier')
# QK(0) from K buffer 0
for r in range(NF):
    v, o = lds_addr(kbuf(0) + kfrag(r) * 1024); e(f'ds_read_b128 {Fr(r)}, {v} offset:{o}')
issued = NF
for i in range(32):
    e(f's_waitcnt lgkmcnt({min(issued - i - 1, 15)})')
    j, d = i & 3, i >> 2
    c = '0' if d == 0 else vr(V_S0 + 16 * j, 16)
    e(f'v_mfma_f32_32x32x16_bf16 {vr(V_S0+16*j, 16)}, {Fr(i)}, {vr(V_Q+4*d, 4)}, {c}')
    if issued < 32:
        v, o = lds_addr(kbuf(0) + kfrag(issued) * 1024); e(f'ds_read_b128 {Fr(issued)}, {v} offset:{o}'); issued += 1
nops(19)
e('s_barrier')                       # all waves done reading K buffer 0 before B's first loads refill it
e(f's_mov_b32 {sr(S_T)}, 0')
e(f's_cmp_lt_u32 {sr(S_WAVE)}, 4')
e('s_cbranch_scc0 .LgrpB')
NU = 6
lab('.LgrpA')
for u in range(NU):
    lab(f'.LA{u}')
    softmax(u, S_T, (lambda: loadsetup_split(S_T, False)) if SPLITLD else (lambda: loadsetup(S_T)))
    if PRIO: e('s_setprio 1')
    e(f's_cmp_eq_u32 {sr(S_T)}, {sr(S_NTM1)}')
    barrier(0)
    e(f's_cbranch_scc1 .LAfin{u}')
    mfma_phase(u, False, False)
    e(f's_add_u32 {sr(S_T)}, {sr(S_T)}, 1')
    barrier(8 if SPLITLD else 4)
e('s_branch .LA0')
for u in range(NU):
    lab(f'.LAfin{u}'); mfma_phase(u, True); barrier(0); e('s_branch .Lepi')
lab('.LgrpB')
softmax(0, S_T, (lambda: loadsetup_split(S_T, True)) if SPLITLD else (lambda: loadsetup(S_T)))
if PRIO: e('s_setprio 1')
e(f's_cmp_eq_u32 {sr(S_T)}, {sr(S_NTM1)}')
for u in range(NU):
    lab(f'.LB{u}')
    e(f's_cbranch_scc1 .LBfin{u}')
    mfma_phase(u, False, True)
    barrier(8 if SPLITLD else 4)
    e(f's_add_u32 {sr(S_TMP2)}, {sr(S_T)}, 1')
    softmax((u + 1) % NU, S_TMP2, (lambda: loadsetup_split(S_TMP2, True)) if SPLITLD else (lambda: loadsetup(S_TMP2)))
    e(f's_add_u32 {sr(S_T)}, {sr(S_T)}, 1')
    if PRIO: e('s_setprio 1')
    e(f's_cmp_eq_u32 {sr(S_T)}, {sr(S_NTM1)}')
    barrier(0)
e('s_branch .LB0')
for u in range(NU):
    lab(f'.LBfin{u}'); mfma_phase(u, True); barrier(0); e('s_barrier'); e('s_branch .Lepi')
L.extend(OOLL)
exec(open(''+__import__('os').path.dirname(__import__('os').path.abspath(__file__))+'/fa18_epi.inc').read().replace('131072', '163840').replace("{sr(S_N)}, {vr(V_ROW)}", "{sr(S_NQ)}, {vr(V_ROW)}").replace("s_mul_i32 {sr(S_TMP)}, {sr(S_N)}, s3", "s_mul_i32 {sr(S_TMP)}, {sr(S_NQ)}, s3"))
open(sys.argv[1], 'w').write('\n'.join(L) + '\n')
print(f'{sys.argv[1]}: {len(L)} lines')
