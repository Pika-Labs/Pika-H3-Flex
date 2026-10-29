# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Mellis, Inc.
"""Local audio decoding for model references."""
from pathlib import Path
import numpy as np

def decode_audio_stereo(path: Path, sampling_rate: int = 32000, max_seconds: float = 16.0) -> np.ndarray | None:
    """The clip's soundtrack as `(2, num_samples)` float32 at `sampling_rate`, or `None` if it has none.

    PyAV's resampler does the rate conversion and the channel-layout downmix in one pass — a mono source is
    upmixed to stereo, a 5.1 source is downmixed layout-aware so the centre channel (which carries the
    dialogue) survives. The pipeline's `MiniMaxH3Ref2VASetupStep` resamples with torchaudio instead, which
    is not in the H3 image; both go through a windowed-sinc resampler.
    """
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="fltp", layout="stereo", rate=sampling_rate)
        chunks: list[np.ndarray] = []
        samples = 0
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunk = resampled.to_ndarray()
                chunks.append(chunk)
                samples += chunk.shape[-1]
            if samples >= max_seconds * sampling_rate:
                break
        for resampled in resampler.resample(None):
            chunks.append(resampled.to_ndarray())

    if not chunks:
        return None
    waveform = np.concatenate(chunks, axis=-1).astype(np.float32, copy=False)
    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise ValueError(f"Expected a (2, num_samples) waveform from {path}, got {waveform.shape}")
    if waveform.shape[1] == 0:
        return None
    return waveform[:, :round(max_seconds * sampling_rate)]


def decode_video_reference(path, max_seconds=16.0):
    """Decode a local video with its real FPS and 32-kHz stereo soundtrack.

    Bound decode duration before allocating frames. Request normalization further
    truncates both modalities to the generated clip duration.
    """
    import av
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.base_rate or 0)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Reference video has no valid frame rate")
        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= int(np.ceil(max_seconds * fps)):
                break
    if not frames:
        raise ValueError("Reference video contains no decoded frames")
    return np.stack(frames), fps, decode_audio_stereo(path, max_seconds=max_seconds)


def preflight_reference_specs(specs):
    """Validate counts and source durations from headers before decoding media.

    Each video/audio is 2–15 seconds; video and standalone-audio totals are
    independently at most 15 seconds. Embedded video soundtracks are part of
    their video reference, not an additional standalone audio reference.
    """
    import av
    counts = {"image": 0, "video": 0, "audio": 0}
    parsed = []
    for spec in specs:
        kind, separator, path = spec.partition(":")
        if not separator or kind not in counts or not path:
            raise ValueError("References must be image:path, video:path, or audio:path")
        counts[kind] += 1
        parsed.append((kind, path))
    if sum(counts.values()) > 12 or counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3:
        raise ValueError("Reference limits: 9 images, 3 videos, 3 audio; 12 total")
    totals = {"video": 0.0, "audio": 0.0}
    for kind, path in parsed:
        if kind == "image":
            continue
        with av.open(path) as container:
            streams = container.streams.video if kind == "video" else container.streams.audio
            if not streams:
                raise ValueError(f"Reference has no {kind} stream: {path}")
            stream = streams[0]
            duration = (float(stream.duration * stream.time_base) if stream.duration is not None and stream.time_base
                        else float(container.duration / av.time_base) if container.duration is not None else None)
            if duration is None or not np.isfinite(duration) or not 2.0 <= duration <= 15.0 + 1e-6:
                raise ValueError(f"Each {kind} reference must have a known duration between 2 and 15 seconds: {path}")
            totals[kind] += duration
    if any(duration > 15.0 + 1e-6 for duration in totals.values()):
        raise ValueError("Total video-reference and total audio-reference durations must each be at most 15 seconds")
    return {"counts": counts, "duration_totals": totals}
