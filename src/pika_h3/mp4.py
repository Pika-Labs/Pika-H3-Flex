# SPDX-License-Identifier: Apache-2.0
"""MP4 writing for the multi-GPU worker.

``write_mp4`` uses the libx264 / AAC settings of the reference writer ``core.vae.write_mp4`` (frame+slice threading enabled).
The worker's segmented path does NOT: each rank encodes its frames as a video-only segment (``encode_segment`` /
``encode_segment_yuv``) with libx264 ``preset=ultrafast``, ``crf=16`` and ``bframes=0`` (overridable with ``PIKA_X264_PRESET`` /
``PIKA_X264_CRF``), optionally from frames converted to yuv420p on the GPU (``rgb_to_yuv420p``, BT.601 limited range, the
conversion swscale applies), and ``mux_segments`` concatenates the segments with the AAC audio. These settings are faster and,
measured on decoded frames, closer to the decoded pixels than the reference encoder (42.2 vs 39.6 dB PSNR); the files are larger.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np


def write_mp4(path: str, frames: np.ndarray, fps: int = 24, audio: Optional[np.ndarray] = None, sample_rate: int = 32000) -> None:
    import av

    container = av.open(path, mode="w")
    vstream = container.add_stream("libx264", rate=fps)
    vstream.width = int(frames.shape[2])
    vstream.height = int(frames.shape[1])
    vstream.pix_fmt = "yuv420p"
    vstream.thread_type = "AUTO"
    astream = None
    if audio is not None:
        astream = container.add_stream("aac", rate=sample_rate)
        astream.layout = "stereo"
    for frame in frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        for packet in vstream.encode(vf):
            container.mux(packet)
    for packet in vstream.encode():
        container.mux(packet)
    if astream is not None:
        pcm = np.ascontiguousarray(audio.astype(np.float32))
        chunk = 1024
        for start in range(0, pcm.shape[1], chunk):
            af = av.AudioFrame.from_ndarray(pcm[:, start:start + chunk], format="fltp", layout="stereo")
            af.sample_rate = sample_rate
            af.pts = start
            for packet in astream.encode(af):
                container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)
    container.close()


def encode_segment(path: str, frames: np.ndarray, fps: int = 24, threads: int = 8) -> None:
    """Video-only libx264 segment without B-frames (decode order == display order, so segments concatenate)."""
    import av

    final, path = path, path + ".part"  # atomic: the file appears under its name only when complete (mux prefetch polls)
    container = av.open(path, mode="w", format="mp4")
    v = container.add_stream("libx264", rate=fps)
    v.width, v.height = int(frames.shape[2]), int(frames.shape[1])
    v.pix_fmt = "yuv420p"
    v.thread_type = "AUTO"
    v.codec_context.thread_count = threads
    # No B-frames (segments concatenate). ultrafast + crf 16 encodes a 32-frame 768p segment in ~35 ms instead of ~230 ms (medium,
    # crf 20) and is closer to the decoded frames (42.2 dB vs 40.2 dB PSNR; the default encoder gives 39.6 dB); the file is larger.
    v.options = {"x264-params": "bframes=0", "crf": os.environ.get("PIKA_X264_CRF", "16"),
                 "preset": os.environ.get("PIKA_X264_PRESET", "ultrafast")}
    for frame in frames:
        vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        for packet in v.encode(vf):
            container.mux(packet)
    for packet in v.encode():
        container.mux(packet)
    container.close()
    os.replace(path, final)


def rgb_to_yuv420p(x):
    """uint8 RGB frames [n, H, W, 3] (torch, any device) -> uint8 yuv420p planes [n, H*3//2, W] (Y, then U and V packed as
    rows of width W: the layout of av.VideoFrame.from_ndarray(format="yuv420p")). BT.601 limited range (what swscale uses for
    rgb24 -> yuv420p without colorspace tags), chroma = 2x2 box mean of RGB. Conversion PSNR equal to swscale's (44.67 vs
    44.65 dB on real decoded frames; 43.22 vs 43.22 dB through x264)."""
    import torch
    n, H, W, _ = x.shape
    f = x.float()
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    y = (16.0 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0).round_().clamp_(0, 255)
    p = f.view(n, H // 2, 2, W // 2, 2, 3).mean((2, 4))
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    u = (128.0 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0).round_().clamp_(0, 255)
    v = (128.0 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0).round_().clamp_(0, 255)
    return torch.cat([y.reshape(n, H, W), u.reshape(n, H // 4, W), v.reshape(n, H // 4, W)], 1).to(torch.uint8)


def encode_segment_yuv(path: str, planes: np.ndarray, width: int, height: int, fps: int = 24, threads: int = 8) -> None:
    """encode_segment for frames already converted to yuv420p (planes [n, H*3//2, W] uint8, see rgb_to_yuv420p): no CPU
    colour conversion (x264 alone: 31 ms for 46 frames vs 50-63 ms with rgb24 input)."""
    import av

    final, path = path, path + ".part"
    container = av.open(path, mode="w", format="mp4")
    v = container.add_stream("libx264", rate=fps)
    v.width, v.height = int(width), int(height)
    v.pix_fmt = "yuv420p"
    v.thread_type = "AUTO"
    v.codec_context.thread_count = threads
    v.options = {"x264-params": "bframes=0", "crf": os.environ.get("PIKA_X264_CRF", "16"),
                 "preset": os.environ.get("PIKA_X264_PRESET", "ultrafast")}
    for fr in planes:
        for packet in v.encode(av.VideoFrame.from_ndarray(fr, format="yuv420p")):
            container.mux(packet)
    for packet in v.encode():
        container.mux(packet)
    container.close()
    os.replace(path, final)


def encode_audio(path: str, audio: np.ndarray, sample_rate: int = 32000) -> None:
    """Audio-only MP4 with the AAC settings of ``mux_segments`` (run off the critical path; its packets are copied by the mux)."""
    import av

    final, path = path, path + ".part"
    c = av.open(path, mode="w", format="mp4")
    a = c.add_stream("aac", rate=sample_rate)
    a.layout = "stereo"
    pcm = np.ascontiguousarray(audio.astype(np.float32))
    for st in range(0, pcm.shape[1], 1024):
        af = av.AudioFrame.from_ndarray(np.ascontiguousarray(pcm[:, st:st + 1024]), format="fltp", layout="stereo")
        af.sample_rate = sample_rate
        af.pts = st
        for packet in a.encode(af):
            c.mux(packet)
    for packet in a.encode():
        c.mux(packet)
    c.close()
    os.replace(path, final)


def mux_segments(out_path: str, segments, audio: Optional[np.ndarray], fps: int = 24, sample_rate: int = 32000,
                 audio_file: Optional[str] = None) -> None:
    """segments: [(start_frame, path)] in order. Copies the video packets (timestamps shifted) and adds the AAC track."""
    import av

    out = av.open(out_path, mode="w", format="mp4")
    first = av.open(segments[0][1])
    vout = out.add_stream_from_template(first.streams.video[0])
    first.close()
    astream = None
    ain = None
    if audio_file is not None:  # pre-encoded AAC packets (identical bitstream to encoding here), copied
        ain = av.open(audio_file)
        astream = out.add_stream_from_template(ain.streams.audio[0])
    elif audio is not None:
        astream = out.add_stream("aac", rate=sample_rate)
        astream.layout = "stereo"
    for start, seg in segments:
        inc = av.open(seg)
        vin = inc.streams.video[0]
        off = int(round(start / fps / float(vin.time_base)))
        for pkt in inc.demux(vin):
            if pkt.dts is None:
                continue
            pkt.pts += off
            pkt.dts += off
            pkt.stream = vout
            out.mux(pkt)
        inc.close()
    if ain is not None:
        for pkt in ain.demux(ain.streams.audio[0]):
            if pkt.dts is None:
                continue
            pkt.stream = astream
            out.mux(pkt)
        ain.close()
    elif astream is not None:
        pcm = np.ascontiguousarray(audio.astype(np.float32))
        chunk = 1024
        for s in range(0, pcm.shape[1], chunk):
            af = av.AudioFrame.from_ndarray(pcm[:, s:s + chunk], format="fltp", layout="stereo")
            af.sample_rate = sample_rate
            af.pts = s
            for packet in astream.encode(af):
                out.mux(packet)
        for packet in astream.encode():
            out.mux(packet)
    out.close()


class MuxPrefetch:
    """Rank 0 (PIKA_MUX_PREFETCH=1): open and demux the segment files (and the AAC file) in background threads (one per file) as
    soon as each appears (writers rename complete files into place), so after the final barrier only the in-memory packets are
    muxed (av.open costs ~5 ms per file: 11 opens were 47 of the 57 ms mux at 10 s). Same packets, same order as mux_segments."""

    def __init__(self, segments, audio_file, timeout=120.0):
        import threading
        self.segments, self.audio_file, self.timeout = list(segments), audio_file, timeout
        files = [p for _, p in self.segments] + ([audio_file] if audio_file is not None else [])
        self.res = [None] * len(files)
        self.ths = [threading.Thread(target=self._one, args=(i, f, i == len(self.segments)), daemon=True) for i, f in enumerate(files)]
        for t in self.ths:
            t.start()

    def _one(self, i, path, audio):
        import time
        import av
        try:
            t0 = time.monotonic()
            while not os.path.exists(path):
                if time.monotonic() - t0 > self.timeout:
                    raise TimeoutError(path)
                time.sleep(0.0005)
            c = av.open(path)
            st = c.streams.audio[0] if audio else c.streams.video[0]
            self.res[i] = (c, st, [p for p in c.demux(st) if p.dts is not None])
        except Exception as e:  # noqa: BLE001 (reported to the caller, which falls back to mux_segments)
            self.res[i] = e

    def join(self):
        for t in self.ths:
            t.join()

    def mux(self, out_path, fps: int = 24):
        import av
        self.join()
        for r in self.res:
            if isinstance(r, Exception) or r is None:
                raise r if r is not None else RuntimeError("prefetch incomplete")
        n = len(self.segments)
        out = av.open(out_path, mode="w", format="mp4")
        vout = out.add_stream_from_template(self.res[0][1])
        astream = out.add_stream_from_template(self.res[n][1]) if self.audio_file is not None else None
        for (start, _), (c, vin, pkts) in zip(self.segments, self.res[:n]):
            off = int(round(start / fps / float(vin.time_base)))
            for pkt in pkts:
                pkt.pts += off
                pkt.dts += off
                pkt.stream = vout
                out.mux(pkt)
        if astream is not None:
            for pkt in self.res[n][2]:
                pkt.stream = astream
                out.mux(pkt)
        out.close()
        self.close()

    def close(self):
        for r in self.res:
            if isinstance(r, tuple):
                r[0].close()
