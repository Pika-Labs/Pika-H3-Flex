<p align="center">
  <img src="assets/pika-h3-flex-banner.png" width="1000" alt="Pika H3 Flex" />
</p>
<h1 align="center">Pika H3 Flex</h1>
<p align="center"><strong>Built for professional iteration.</strong></p>
<p align="center">Six-step generation · Omni-reference · Native 480p / 768p · Up to 4K via Pika API</p>
<p align="center">
  <a href="https://dev.pika.art" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Pika%20API-7C3AED" alt="Pika API" /></a>
  <a href="https://create.pika.art" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Create%20with%20Pika-EC4899" alt="Create with Pika" /></a>
  <a href="https://pika.art/blog/pika-h3-flex-research" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Tech%20Blog-2563EB" alt="Tech Blog" /></a>
  <a href="https://github.com/Pika-Labs/Pika-H3-Flex" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white" alt="GitHub" /></a>
  <a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face" /></a>
</p>
<p align="center">
  <a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex">Model weights</a> ·
  <a href="#watch-the-model">Watch samples</a> ·
  <a href="#install">Quickstart</a> ·
  <a href="docs/prompt-guidance.md">Prompt guide</a> ·
  <a href="#pika-api-extensions">Pika API extensions</a>
</p>

Pika H3 Flex is a post-trained H3 model, designed to give professional creators the flexibility and speed to explore with precision. Optimized for greater control, it generates video with synchronized audio in six sampling steps and supports up to 12 reference inputs, with shot direction through references and prompts.

Pika’s hosted API extends this workflow with [1080p, 2K and 4K video generation](https://dev.pika.art/models/pika/pika-h3-flex/draft-to-video), [audio stems](https://dev.pika.art/models/pika/pika-h3-flex/audio-stems) and [prompt enhancement](https://dev.pika.art/models/pika/pika-h3-flex/prompt-rewriter). Local open-weight inference supports native 480p / 768p video with mixed stereo audio.

**Ideal for multi-shot storytelling · Motion graphics · Product films · Reference-guided animation**

## Watch the model

Explore camera choreography, cinematic storytelling, graphic design and reference-guided animation. Click a preview to watch with sound.

<table>
<tr>
<td align="center" width="50%"><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/camera-rally.mp4"><img src="assets/showcases/camera-rally.jpg" alt="Frog table-tennis rally" width="440" /></a><br /><strong>Frog table-tennis rally</strong><br />Camera choreography<br /><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/camera-rally.mp4">▶ Watch with audio</a></td>
<td align="center" width="50%"><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/balcony-orbit.mp4"><img src="assets/showcases/balcony-orbit.jpg" alt="A journey between balconies" width="440" /></a><br /><strong>A journey between balconies</strong><br />Text-to-video camera control<br /><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/balcony-orbit.mp4">▶ Watch with audio</a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/sitcom-story.mp4"><img src="assets/showcases/sitcom-story.jpg" alt="The birthday-cake heist" width="440" /></a><br /><strong>The birthday-cake heist</strong><br />Multi-shot storytelling<br /><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/sitcom-story.mp4">▶ Watch with audio</a></td>
<td align="center" width="50%"><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/motion-graphics.mp4"><img src="assets/showcases/motion-graphics.jpg" alt="MOVE: kinetic typography" width="440" /></a><br /><strong>MOVE: kinetic typography</strong><br />Motion graphics<br /><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/motion-graphics.mp4">▶ Watch with audio</a></td>
</tr>
<tr>
<td align="center" width="50%"><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/product-film.mp4"><img src="assets/showcases/product-film.jpg" alt="Feel the bass" width="440" /></a><br /><strong>Feel the bass</strong><br />Product film<br /><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/product-film.mp4">▶ Watch with audio</a></td>
<td align="center" width="50%"><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/motion-transfer.mp4"><img src="assets/showcases/motion-transfer.jpg" alt="Record-shop dance" width="440" /></a><br /><strong>Record-shop dance</strong><br />Reference-guided motion<br /><a href="https://huggingface.co/Pika-Labs/Pika-H3-Flex/resolve/main/assets/showcases/motion-transfer.mp4">▶ Watch with audio</a></td>
</tr>
</table>

[Prompts and input assets for these six showcases](docs/showcases.md)

These are creative showcases, not release-validation renders or throughput benchmarks. Exact reproduction with the released package is not guaranteed.

## System overview

Pika H3 Flex is one video-and-audio model family with two task adapters: FL2VA for text and endpoint-frame conditioning, and Ref2VA for multimodal references. It builds on [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) and generates video with synchronized stereo audio in six sampling steps.

The inference pipeline encodes the prompt and input media, generates video and audio latents with the selected Pika H3 Flex adapter, then decodes and combines them into an MP4. The package downloads the pinned base components and adapter automatically. Use the [prompt guide](docs/prompt-guidance.md) to describe scenes, motion, dialogue, ambience and music; optional [prompt enhancement](https://dev.pika.art/models/pika/pika-h3-flex/prompt-rewriter) is available through the Pika API.

### Input and output specifications

| Category                 | Specification                                                                                                     |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| Input modalities         | Text; first/last images; image, video and audio references, depending on the selected mode                        |
| Native output resolution | 480p / 768p across all 11 supported aspect ratios; exact pixel dimensions follow the selected aspect-ratio bucket |
| Output duration          | Requested integer durations of 5–15 seconds; actual length follows the model frame grid                          |
| Output aspect ratios     | 21:9, 16:9, 3:2, 4:3, 5:4, 1:1, 4:5, 3:4, 2:3, 9:16, 9:21; dimensions use the packaged resolution buckets         |
| Output frame rate        | 24 fps                                                                                                            |
| Output audio             | 32 kHz stereo, mixed into the generated video                                                                     |
| Output files             | MP4 with H.264 video and AAC audio, plus JSON generation metadata                                                 |
| Prompt input             | Text or a UTF-8 prompt file; 1–8,000 characters                                                                  |
| Sampling                 | Six generator evaluations using the released adapter's sampling schedule                                          |

Durations are rounded up to the supported frame grid: a 5-second request produces 124 frames (about 5.17 seconds), and a 15-second request produces 362 frames (about 15.08 seconds). Native resolution and aspect ratio jointly determine the actual canvas dimensions.

### Generation modes and reference limits

| Mode                              | Inputs                                                                      | Adapter / CLI task  |
| --------------------------------- | --------------------------------------------------------------------------- | ------------------- |
| Text to video and audio           | Prompt, without input media                                                 | FL2VA / `t2va`    |
| First-frame conditioning          | Prompt and one opening frame                                                | FL2VA / `i2va`    |
| Last-frame conditioning           | Prompt and one ending frame                                                 | FL2VA / `l2va`    |
| First-and-last-frame conditioning | Prompt and both endpoint frames                                             | FL2VA / `fl2va`   |
| Multimodal references             | Prompt and ordered image/video references, optionally with audio references | Ref2VA / `ref2va` |

Ref2VA accepts up to **9 images, 3 videos and 3 standalone audio clips**, with **12 reference files total**. At least one image or video is required. Each video or standalone audio clip must be **2–15 seconds**; combined video duration and combined standalone-audio duration must each be at most **15 seconds**. A video's embedded soundtrack stays with that video and does not count as an additional standalone audio file.

These are accepted input limits, not a guarantee that every maximum-size combination fits GPU memory. The package runs one request on one GPU, with optional CPU offload. See [hardware requirements and tested configurations](docs/hardware.md) for representative H100 results and memory guidance.

## Pika API extensions

Extend your generation workflow with Pika’s hosted endpoints:

| Feature                 | Support                                            | Endpoint                                                                           |
| ----------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Higher-resolution video | Generate 1080p, 2K and 4K video from a draft video | [Draft to video](https://dev.pika.art/models/pika/pika-h3-flex/draft-to-video)      |
| Audio stems             | Obtain separate audio stems                        | [Audio stems](https://dev.pika.art/models/pika/pika-h3-flex/audio-stems)            |
| Prompt enhancement      | Enhance generation prompts                         | [Prompt enhancement](https://dev.pika.art/models/pika/pika-h3-flex/prompt-rewriter) |

These optional hosted services are separate from the downloadable adapters and local inference package. The specifications above describe local model inference; see each endpoint’s documentation for its API capabilities and usage.

## Install

The command and Python distribution are named `pika-h3-pro`.

Clone the repository, then enter its directory:

```bash
git clone https://github.com/Pika-Labs/Pika-H3-Flex.git
cd Pika-H3-Flex
```

For pip installation, install system `git`, `patch`, `ffmpeg` and a C compiler with development headers first (for example, `build-essential` on Debian/Ubuntu). Triton needs the compiler at runtime. Docker and Conda recipes supply these prerequisites.

Use Python 3.12, an NVIDIA GPU and a compatible driver. Representative cases are tested on one H100 80GB per request; H200 and B300 remain untested. Longer clips and large reference sets can exceed GPU memory. Downloading both tasks acquired approximately 210 GB of base files plus 5.5 GB of adapters; allow additional space for dependencies and download staging. See [tested cases and memory measurements](docs/hardware.md).

**GPU selection:** the current package runs one request on one GPU. The examples expose a single GPU with `--gpus 'device=0'`; exposing eight GPUs does not shard or accelerate a request. Multiple idle GPUs can run independent processes. Start qualification at 480p / 5 seconds. Multi-GPU inference within one request is not implemented in this package.

```bash
python -m pip install -r requirements.txt
python tools/apply_diffusers_patches.py
python -m pip install --no-deps -e .
pika-h3-pro --help
```

Alternatively, create the Conda environment, then install the package:

```bash
conda env create -f environment.yml
conda activate pika-h3-pro
python tools/apply_diffusers_patches.py
python -m pip install --no-deps -e .
```

For Docker:

```bash
docker build -t pika-h3-pro .
docker run --rm --gpus 'device=0' pika-h3-pro --help
```

The three recipes share `requirements.txt`. Authentication, when required, is passed at runtime using `HF_TOKEN` or the Hugging Face cache; never put tokens into image layers.

## Generate

The released adapter pair and immutable HF pin are recorded in [release.json](release.json). Use the pinned release with:

```bash
export MODEL_REPO=Pika-Labs/Pika-H3-Flex
export MODEL_REVISION=0ddf55ad7067ca1e76812253bd7b4600df87f759
```

```bash
pika-h3-pro --task t2va --prompt-file examples/prompts/t2va.txt \
  --model-repo "$MODEL_REPO" --revision "$MODEL_REVISION" \
  --resolution 480p --aspect 16:9 --duration 5 --seed 42 --output outputs/sample.mp4
```

Use `--task i2va --first-image first.png`, `--task l2va --last-image last.png`, or `--task fl2va --first-image first.png --last-image last.png` for endpoint conditioning. Ref2VA accepts ordered `--ref image:subject.png` inputs with a reference prompt:

```bash
pika-h3-pro --task ref2va --prompt-file examples/prompts/ref2va.txt \
  --ref image:subject.png --model-repo "$MODEL_REPO" --revision "$MODEL_REVISION" \
  --resolution 480p --duration 5 --output outputs/reference.mp4
```

The downloader acquires the selected adapter and its pinned MiniMax-H3 components. Local snapshot overrides exist for validation, but are not required by the public workflow. The package validates adapter hashes and sampling parameters before loading the transformer. Adapter files remain FP32; resident LoRA computation uses BF16, preserving native mixed-precision model boundaries.

See [prompt guidance](docs/prompt-guidance.md). Prompt files are read verbatim; no prompt rewriting service is required. The output JSON records the final prompt, seed, artifact identity, geometry, offload setting and timings. Do not publish a metadata file containing a private prompt without reviewing it.

For requests that exceed GPU memory, add `--cpu-offload`. This keeps transformer weights on the CPU and prefetches blocks to the GPU; it trades substantial host RAM and additional transfer time for lower sampling VRAM. It does not offload the text encoder or VAEs, so the reduced sampling peak is not a minimum GPU requirement. The default keeps the transformer on the GPU. Consult the verification table before choosing clip length and resolution.

For example, this 768p/15s configuration passed on a single H100 80GB with offload:

```bash
pika-h3-pro --model-repo "$MODEL_REPO" --revision "$MODEL_REVISION" \
  --task t2va --prompt-file examples/prompts/t2va.txt --resolution 768p \
  --duration 15 --seed 42 --cpu-offload --output outputs/long-768p.mp4
```

Requested durations are 5–15 integer seconds. The model uses a 17n+5 frame grid at 24fps, so actual duration differs slightly from the request. The [verification table](docs/hardware.md) records tested cases; it does not qualify every aspect/duration/reference combination. Raw video and audio references use `--ref video:clip.mp4` and `--ref audio:sound.wav`. Video soundtracks are included automatically. Limits: 9 images, 3 videos, 3 audio references, 12 references total. Each video/audio source must be 2–15 seconds; video durations and standalone audio durations separately sum to at most 15 seconds. Input order is preserved. Mixed image/video/audio input has passed a 480p/5s H100 test; maximum reference counts remain unqualified on GPU.

## Reproducibility

Both adapters use six denoising forwards with knots 1000/750/500/250/100/40/0, video shift12 and audio shift3, with fresh intermediate noise. Generic six-step Euler sampling is not equivalent. Results may differ across attention kernels, devices and library versions even with identical seeds.

Model components are provided by MiniMax-H3, Diffusers and Transformers; adapter injection uses PEFT.

[Hardware requirements and tested configurations](docs/hardware.md)

## License

These adapters are Model Derivatives of MiniMax H3 and are distributed under the
[MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/42ed227ee7df40d41602854ae760620d6eb651fe/LICENSE).

See [NOTICE](NOTICE) for the required MiniMax notice and Mellis, Inc. copyright statement.

## Attribution

If you use these adapters in research or a product, we'd appreciate a credit to Pika
and a citation:

```bibtex
@misc{pika2026flex,
  title  = {Pika H3 Flex Technical Deep Dive},
  author = {{Pika Research}},
  year   = {2026},
  howpublished = {\url{https://pika.art/blog/pika-h3-flex-research}}
}
```

## Code license

The inference and loader code is licensed under [Apache-2.0](LICENSE-CODE). The model license applies to the adapters. Upstream code attribution is retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
