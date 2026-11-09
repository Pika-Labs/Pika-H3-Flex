# SPDX-License-Identifier: Apache-2.0
"""Tile-parallel VAE decode: the released clip/tile decomposition of ``AutoencoderKLMiniMaxH3._decode`` distributed
over all ranks (same tiles, same blending, same arithmetic; the work is only partitioned).

Units = (clip, tile). Contiguous unit ranges are assigned to ranks; the rank holding most tiles of a clip owns it,
stitches it, blends the temporal overlap with the previous clip and converts to uint8; owners send frames to rank 0.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def _a2a_rows(rows_by_dest, row_shape, dtype, dev, in_counts, out_counts, group):
    """Collective row exchange. rows_by_dest[d]: list of tensors (each row_shape) sent to rank d;
    in_counts[d] / out_counts[s]: number of rows sent to d / received from s (known to every rank)."""
    numel = 1
    for x in row_shape:
        numel *= x
    flat = []
    for d in range(len(in_counts)):
        flat += [t.reshape(1, numel) for t in rows_by_dest[d]]
    send = torch.cat(flat, 0) if flat else torch.empty((0, numel), dtype=dtype, device=dev)
    recv = torch.empty((sum(out_counts), numel), dtype=dtype, device=dev)
    dist.all_to_all_single(recv, send, output_split_sizes=list(out_counts), input_split_sizes=list(in_counts), group=group)
    out, i = [], 0
    for s_ in range(len(out_counts)):
        out.append([recv[i + j].view(row_shape) for j in range(out_counts[s_])])
        i += out_counts[s_]
    return out


class TileDecoder:
    def __init__(self, decoders, rank, world, group=None):
        self.dec = decoders  # H3Decoders (video VAE on every rank)
        self.rank, self.world, self.group = rank, world, group
        # max tiles per decoder call; the rank's tiles are split into equal batches <= this (tile bench, GPU 0: 13.3 ms/tile at B=4,
        # 12.5 at B=6, 12.25 at B=12: larger GEMMs)
        self.batch = int(os.environ.get("PIKA_TILE_BATCH", "12"))
        # The VAE decode runs in bf16 by default: bf16 tile decoder (FastViTDecoder,
        # ~55 dB tile SNR on real latents vs the 45 dB gate, 15.5 ms/tile). PIKA_FAST_VIT=0 selects the reference-precision
        # decoder (PIKA_VAE_GEMM=f16x3 default: fp16 3-term split GEMMs, 47.7 ms/tile; =fp32 plain; =x6 bf16x6).
        from .fast_vit import ExactViTDecoder, FastViTDecoder
        mode = os.environ.get("PIKA_VAE_GEMM", "f16x3")
        if os.environ.get("PIKA_FAST_VIT", "1") == "1":
            self.fast = FastViTDecoder(decoders.vae.decoder)
            old_blocks = decoders.vae.decoder.transformer_blocks
            decoders.vae.decoder.transformer_blocks = torch.nn.ModuleList()  # fp32 blocks are not needed any more
            for prm in old_blocks.parameters():
                prm.data = torch.empty(0, device=prm.device)
            del old_blocks
        else:
            self.fast = ExactViTDecoder(decoders.vae.decoder, mode)
        # post_quant_conv is a pointwise (1x1x1) conv: run it as one fp32 linear over the whole latent video instead of a MIOpen
        # conv per tile batch (MIOpen's per-call solver search with a null workspace was followed by hard crashes in the HSA runtime).
        pq = decoders.vae.post_quant_conv
        self.pq_w = pq.weight.detach().reshape(pq.out_channels, pq.in_channels).clone().float()
        self.pq_b = pq.bias.detach().clone().float()
        decoders.vae.encoder = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def plan(self, F_lat, H_lat, W_lat, weights=None):
        vae = self.dec.vae
        ratio = vae.spatial_compression_ratio
        height, width = H_lat * ratio, W_lat * ratio
        th, tw, oh, ow = (vae.tile_sample_min_height, vae.tile_sample_min_width, vae.tile_sample_min_overlap_height,
                          vae.tile_sample_min_overlap_width)
        from . import lossy as LS
        vt = LS.get("vae_tile") if LS.ACTIVE else None  # LOSSY (off by default): [tile_h, tile_w, min_overlap_h, min_overlap_w] px
        if vt:
            th, tw, oh, ow = (int(x) for x in vt)
            assert all(x % ratio == 0 for x in (th, tw, oh, ow)), "vae_tile values must be multiples of the latent stride"
        yi, yl, yo = vae._split_tiles(height, th, oh)
        xi, xl, xo = vae._split_tiles(width, tw, ow)
        tcs = vae.tokens_chunk_size
        token_drop = vae.config.token_drop
        num_tokens = F_lat + token_drop
        assert (-num_tokens) % tcs == 0 and token_drop > 0, "unsupported temporal geometry for parallel decode"
        n_clips = (num_tokens // tcs) - 1
        clip_len = tcs + vae.token_overlap
        T = len(yi) * len(xi)
        U = n_clips * T
        w = self.world
        weights = weights or [1.0] * w
        tot = sum(weights)
        cum = [0.0]
        for x in weights:
            cum.append(cum[-1] + x / tot * U)
        bounds = [int(round(c)) for c in cum]
        bounds[-1] = U
        unit_rank = []
        for r in range(w):
            unit_rank += [r] * (bounds[r + 1] - bounds[r])
        owner = []
        for c in range(n_clips):
            cnt = {}
            for t in range(T):
                cnt[unit_rank[c * T + t]] = cnt.get(unit_rank[c * T + t], 0) + 1
            owner.append(max(sorted(cnt), key=lambda r: cnt[r]))
        return dict(yi=yi, yl=yl, yo=yo, xi=xi, xl=xl, xo=xo, ratio=ratio, n_clips=n_clips, T=T, unit_rank=unit_rank,
                    owner=owner, clip_len=clip_len, tcs=tcs, height=height, width=width)

    @torch.no_grad()
    def decode(self, latents, weights=None, gather=True):
        """latents: [F, 24, H, W] normalized video latents (identical on all ranks). Returns uint8 [N, H, W, 3] on rank 0
        (None elsewhere)."""
        dec, vae = self.dec, self.dec.vae
        from . import lossy as LS
        if LS.ACTIVE and int(LS.get("vae_clips", 1) or 1) > 1:  # LOSSY (off by default): k temporal clips per decoder call
            return self.decode_grouped(latents, int(LS.get("vae_clips")), weights, gather)
        dev = latents.device
        rank = self.rank
        F_lat, _, H_lat, W_lat = latents.shape
        P = self.plan(F_lat, H_lat, W_lat, weights)
        lat = latents.to(dev, torch.float32).permute(1, 0, 2, 3).unsqueeze(0)
        lat = (lat * dec.video_std + dec.video_mean).to(dec.dtype)  # [1, 24, F, H, W]
        T, nx = P["T"], len(P["xi"])
        ratio = P["ratio"]
        unit_rank, owner = P["unit_rank"], P["owner"]
        torch.backends.cuda.matmul.allow_tf32 = False
        lat = torch.nn.functional.linear(lat.permute(0, 2, 3, 4, 1), self.pq_w, self.pq_b).permute(0, 4, 1, 2, 3).contiguous()
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = (self.fast is not None and not hasattr(self.fast, 'mode'))
        tiles = {}
        _tm = os.environ.get("PIKA_TIMING") == "3"  # dev: per-phase decode times (synchronizing)
        if _tm:
            import time as _time
            torch.cuda.synchronize(); _t = [_time.monotonic()]
            def _mark():
                torch.cuda.synchronize(); _t.append(_time.monotonic())
        mine = [u for u, r in enumerate(unit_rank) if r == rank]
        nb = max(1, -(-len(mine) // self.batch))
        bsz = max(1, -(-len(mine) // nb))  # equal batches (no small tail batch)
        if True:
            for i0 in range(0, len(mine), bsz):
                batch_units = mine[i0:i0 + bsz]
                zs = []
                for u in batch_units:
                    c, t = divmod(u, T)
                    ty, tx = divmod(t, nx)
                    z = lat[:, :, c * P["tcs"]: c * P["tcs"] + P["clip_len"]]
                    ip, il = P["yi"][ty], P["yl"][ty]
                    jp, jl = P["xi"][tx], P["xl"][tx]
                    zs.append(z[..., ip // ratio: ip // ratio + il // ratio, jp // ratio: jp // ratio + jl // ratio])
                if os.environ.get("PIKA_DBG"):
                    fr, tot = torch.cuda.mem_get_info()
                    print(f"[dbg] rank {rank} batch {i0} zs {tuple(zs[0].shape)} free {fr/2**30:.1f} GiB reserved {torch.cuda.memory_reserved()/2**30:.1f} alloc {torch.cuda.memory_allocated()/2**30:.1f}", flush=True)
                zb = torch.cat(zs, 0)
                out = self.fast(zb) if self.fast is not None else vae.decoder(zb)
                for j, u in enumerate(batch_units):
                    tiles[u] = out[j:j + 1]
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        if _tm:
            _mark()
        # ---- exchange tiles to clip owners (collective all-to-all; every rank derives the same counts from the plan)
        W = self.world
        frames_per_clip = P["clip_len"] * vae.temporal_compression_ratio
        tshape = (1, 3, frames_per_clip, P["yl"][0], P["xl"][0])
        send_units = [[] for _ in range(W)]
        recv_units = [[] for _ in range(W)]
        for u, r in enumerate(unit_rank):
            o = owner[u // T]
            if r == o:
                continue
            if r == rank:
                send_units[o].append(u)
            if o == rank:
                recv_units[r].append(u)
        in_counts = [len(x) for x in send_units]
        out_counts = [len(x) for x in recv_units]
        total_counts = sum(1 for u, r in enumerate(unit_rank) if r != owner[u // T])
        if total_counts:
            got = _a2a_rows([[tiles[u] for u in send_units[d]] for d in range(W)], tshape, torch.float32, dev,
                            in_counts, out_counts, self.group)
            for r_ in range(W):
                for u, t_ in zip(recv_units[r_], got[r_]):
                    tiles[u] = t_
        for u in [u for ul in send_units for u in ul]:
            tiles.pop(u, None)
        if _tm:
            _mark()
        # ---- stitch owned clips, split into main frames / overlap tail
        pre = vae.frame_pre_padding
        cnf = P["tcs"] * vae.temporal_compression_ratio
        main, tail = {}, {}
        for c in range(P["n_clips"]):
            if owner[c] != rank:
                continue
            rows = []
            for ty in range(len(P["yi"])):
                rows.append([tiles[c * T + ty * nx + tx] for tx in range(nx)])
            clip = vae._stitch_tiles(rows, P["yo"], P["xo"])
            main[c] = clip[:, :, 0:cnf][:, :, pre:]
            tail[c] = clip[:, :, cnf: 2 * cnf][:, :, pre:]
        del tiles
        if _tm:
            _mark()
        # ---- overlap tails to the owner of the next clip (collective all-to-all)
        tail_frames = frames_per_clip - cnf - pre
        tail_shape = (1, 3, tail_frames, P["height"], P["width"])
        send_t = [[] for _ in range(W)]
        in_c = [0] * W
        out_c = [0] * W
        recv_order = []
        for c in range(P["n_clips"] - 1):
            if owner[c] == owner[c + 1]:
                continue
            if owner[c] == rank:
                send_t[owner[c + 1]].append(tail[c])
                in_c[owner[c + 1]] += 1
            if owner[c + 1] == rank:
                out_c[owner[c]] += 1
                recv_order.append((owner[c], c))
        got = {}
        if any(owner[c] != owner[c + 1] for c in range(P["n_clips"] - 1)):
            res = _a2a_rows(send_t, tail_shape, torch.float32, dev, in_c, out_c, self.group)
            idx = [0] * W
            for (src, c) in recv_order:
                got[c] = res[src][idx[src]]
                idx[src] += 1
        mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(3, 1, 1, 1)

        def to_u8(x):  # x: [1, 3, n, H, W] fp32 -> [n, H, W, 3] uint8
            x = x[0].float() * std + mean
            return x.clamp(0, 1).permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).contiguous()

        if _tm:
            _mark()
        out_chunks = {}
        for c in range(P["n_clips"]):
            if owner[c] != rank:
                continue
            chunk = main[c]
            if c > 0:
                prev = tail[c - 1] if owner[c - 1] == rank else got[c - 1]
                chunk = vae._blend(prev, chunk, vae.frame_overlap, dim=-3)
            out_chunks[c] = to_u8(chunk)
        last = P["n_clips"] - 1
        final_tail = to_u8(tail[last]) if owner[last] == rank else None
        n_main = (P["tcs"] * vae.temporal_compression_ratio - pre)  # frames per clip (17)
        total = n_main * P["n_clips"] + tail_frames
        # frames owned by this rank as one contiguous segment (owners hold consecutive clips)
        parts = [(c * n_main, out_chunks[c]) for c in sorted(out_chunks)]
        if final_tail is not None:
            parts.append((n_main * P["n_clips"], final_tail))
        if parts:
            assert all(parts[i][0] + parts[i][1].shape[0] == parts[i + 1][0] for i in range(len(parts) - 1))
            self.segment = (parts[0][0], torch.cat([x for _, x in parts], 0) if len(parts) > 1 else parts[0][1])
        else:
            self.segment = None
        self.total_frames = total
        # (start frame, rank) of every rank's segment, in frame order (rank 0 prefetches the segment files for the mux)
        firsts = {}
        for c in range(P["n_clips"]):
            firsts.setdefault(owner[c], c)
        self.seg_plan = sorted((c * n_main, r) for r, c in firsts.items())
        if _tm:
            _mark()
            print(f"[dtime] rank {rank} units {len(mine)} owned {sum(1 for o in owner if o == rank)} tiles/xchg/stitch/tails/u8 "
                  + " ".join(f"{(b - a) * 1e3:.0f}" for a, b in zip(_t, _t[1:])) + " ms", flush=True)
        if not gather:
            return None
        # ---- gather frames on rank 0
        per = out_chunks[next(iter(out_chunks))].shape[0] if out_chunks else None
        result = None
        rows_by_dest = [[] for _ in range(W)]
        in_c = [0] * W
        out_c = [0] * W
        for c in range(P["n_clips"]):
            if owner[c] == rank and rank != 0:
                rows_by_dest[0].append(out_chunks[c])
                in_c[0] += 1
            if rank == 0 and owner[c] != 0:
                out_c[owner[c]] += 1
        res = _a2a_rows(rows_by_dest, (n_main, P["height"], P["width"], 3), torch.uint8, dev, in_c, out_c, self.group)
        rows_t = [[] for _ in range(W)]
        in_t = [0] * W
        out_t = [0] * W
        if owner[last] != 0:
            if rank == owner[last]:
                rows_t[0] = [final_tail]
                in_t[0] = 1
            if rank == 0:
                out_t[owner[last]] = 1
        res_t = _a2a_rows(rows_t, (tail_frames, P["height"], P["width"], 3), torch.uint8, dev, in_t, out_t, self.group)
        if rank == 0:
            result = torch.empty((total, P["height"], P["width"], 3), dtype=torch.uint8, device=dev)
            idx = [0] * W
            for c in range(P["n_clips"]):
                if owner[c] == 0:
                    result[c * n_main:(c + 1) * n_main] = out_chunks[c]
                else:
                    result[c * n_main:(c + 1) * n_main] = res[owner[c]][idx[owner[c]]]
                    idx[owner[c]] += 1
            if owner[last] == 0:
                result[n_main * P["n_clips"]:] = final_tail
            else:
                result[n_main * P["n_clips"]:] = res_t[owner[last]][0]
        return result


    @torch.no_grad()
    def _decode_grouped_impl(self, latents, k, weights, gather):
        """LOSSY dev path (lossy key "vae_clips": k): decode k consecutive temporal clips in ONE decoder call (5k + 2 latent tokens
        instead of k x 7: removes the 2-token clip overlap inside a group; frames of a token map as in the clip path, the 3 pre-padding
        frames of an inner clip's first token are dropped, groups are cross-faded as clips are). Gathers on every rank's owner of
        the group, then rank 0 (simple: all frames are sent to rank 0, segments written by rank 0 only)."""
        dec, vae = self.dec, self.dec.vae
        dev = latents.device
        rank, W = self.rank, self.world
        F_lat, _, H_lat, W_lat = latents.shape
        P = self.plan(F_lat, H_lat, W_lat, None)
        n_clips, tcs, T, nx, ratio = P["n_clips"], P["tcs"], P["T"], len(P["xi"]), P["ratio"]
        ovl = P["clip_len"] - tcs
        groups = [list(range(i, min(n_clips, i + k))) for i in range(0, n_clips, k)]
        U = len(groups) * T
        wts = weights or [1.0] * W
        cost = [len(groups[u // T]) * tcs + ovl for u in range(U)]  # tokens per unit ~ cost
        tot = sum(cost)
        tw = sum(wts)
        unit_rank, acc, r = [], 0.0, 0
        for u in range(U):
            while r < W - 1 and acc >= sum(wts[:r + 1]) / tw * tot:
                r += 1
            unit_rank.append(r)
            acc += cost[u]
        lat = latents.to(dev, torch.float32).permute(1, 0, 2, 3).unsqueeze(0)
        lat = (lat * dec.video_std + dec.video_mean).to(dec.dtype)
        lat = torch.nn.functional.linear(lat.permute(0, 2, 3, 4, 1), self.pq_w, self.pq_b).permute(0, 4, 1, 2, 3).contiguous()
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = (self.fast is not None and not hasattr(self.fast, 'mode'))
        maxf = (k * tcs + ovl) * vae.temporal_compression_ratio
        tshape = (1, 3, maxf, P["yl"][0], P["xl"][0])
        mine = [u for u in range(U) if unit_rank[u] == rank]
        tiles = {}
        import time as _time
        torch.cuda.synchronize(); _t0 = _time.monotonic()
        bs = max(1, int(os.environ.get("PIKA_TILE_BATCH", "12")) // k)
        by_len = {}
        for u in mine:
            by_len.setdefault(len(groups[u // T]), []).append(u)
        for _, us in sorted(by_len.items()):
            for i0 in range(0, len(us), bs):  # batches of units of equal group length
                zs = []
                for u in us[i0:i0 + bs]:
                    g, t = divmod(u, T)
                    ty, tx = divmod(t, nx)
                    c0 = groups[g][0]
                    z = lat[:, :, c0 * tcs: c0 * tcs + len(groups[g]) * tcs + ovl]
                    ip, il, jp, jl = P["yi"][ty], P["yl"][ty], P["xi"][tx], P["xl"][tx]
                    zs.append(z[..., ip // ratio: (ip + il) // ratio, jp // ratio: (jp + jl) // ratio])
                o = self.fast(torch.cat(zs, 0))
                for j, u in enumerate(us[i0:i0 + bs]):
                    pad = torch.zeros(tshape, dtype=torch.float32, device=dev)
                    pad[:, :, :o.shape[2]] = o[j:j + 1]
                    tiles[u] = pad
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        torch.cuda.synchronize()
        print(f"[dtime-grouped] rank {rank} k {k} units {len(mine)} tokens {sum(cost[u] for u in mine)} tiles "
              f"{(_time.monotonic() - _t0) * 1e3:.0f} ms", flush=True)
        # all tiles to rank 0 (dev simplicity)
        in_counts = [len(mine) if d == 0 else 0 for d in range(W)]
        out_counts = [sum(1 for u in range(U) if unit_rank[u] == s_) if rank == 0 else 0 for s_ in range(W)]
        if rank == 0:
            in_counts[0] = 0
            out_counts[0] = 0
        send = [[tiles[u] for u in mine] if (d == 0 and rank != 0) else [] for d in range(W)]
        got = _a2a_rows(send, tshape, torch.float32, dev, in_counts, out_counts, self.group)
        self.total_frames = None
        if rank != 0:
            self.segment = None
            self.seg_plan = [(0, 0)]
            return None
        for s_ in range(1, W):
            for u, t_ in zip([u for u in range(U) if unit_rank[u] == s_], got[s_]):
                tiles[u] = t_
        pre = vae.frame_pre_padding
        cnf = tcs * vae.temporal_compression_ratio
        mean = torch.tensor((0.485, 0.456, 0.406), device=dev).view(3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=dev).view(3, 1, 1, 1)
        outs, prev_tail = [], None
        for g, cl in enumerate(groups):
            nf = (len(cl) * tcs + ovl) * vae.temporal_compression_ratio
            rows = [[tiles[g * T + ty * nx + tx][:, :, :nf] for tx in range(nx)] for ty in range(len(P["yi"]))]
            clip = vae._stitch_tiles(rows, P["yo"], P["xo"])
            for j in range(len(cl)):
                m = clip[:, :, j * cnf:(j + 1) * cnf][:, :, pre:]
                if j == 0 and prev_tail is not None:
                    m = vae._blend(prev_tail, m, vae.frame_overlap, dim=-3)
                outs.append(m)
            prev_tail = clip[:, :, len(cl) * cnf:(len(cl) + 1) * cnf][:, :, pre:]
        outs.append(prev_tail)
        x = torch.cat(outs, 2)[0].float() * std + mean
        pix = x.clamp(0, 1).permute(1, 2, 3, 0).mul(255).round().to(torch.uint8).contiguous()
        self.segment = (0, pix)
        self.seg_plan = [(0, 0)]
        self.total_frames = pix.shape[0]
        return pix

    def decode_grouped(self, latents, k, weights=None, gather=True):
        return self._decode_grouped_impl(latents, k, weights, gather)
