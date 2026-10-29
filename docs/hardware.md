# Hardware and tested configurations

This package uses one GPU per request. Additional GPUs can run independent requests; this package does not shard one request across GPUs.

## Tested requests

The following representative cases passed on H100 80GB. Coverage does not establish support for every input combination or maximum reference count.

| Task | Resolution | Requested duration | Memory mode |
| --- | --- | --- | --- |
| Text generation | 480p | 5 / 15 seconds | Default |
| First, last, or both endpoint frames | 480p | 5 seconds | Default |
| Text, image, or both endpoint frames | 768p | 5 seconds | Default |
| Image references | 480p / 768p | 5 seconds | Default |
| Image + video + audio references | 480p | 5 seconds | Default |
| Text or image references | 768p | 15 seconds | CPU offload |

Square and portrait text generation have also passed representative checks. Encoded outputs are checked against the packaged aspect-ratio buckets, expected frame counts, 24 fps and 32 kHz stereo audio. H200 and B300 have not been qualified for this package.

## Memory and performance

Default 480p/5s sampling allocated approximately 66–69 GiB in representative tests. Default 768p/15s exceeded H100 memory; use `--cpu-offload` for that shape.

CPU offload streams transformer blocks from pinned host memory. Representative 768p/15s sampling allocated approximately 29–32 GiB on GPU, with observed host memory near 94 GiB including page cache. Leave additional headroom for references, longer prompts, dependencies and output buffers. These measurements are not minimum system requirements. The text encoder and VAEs have separate memory requirements, and sampling allocation excludes those stages.

Allow at least 211 GB for base components plus approximately 5.5 GB for both adapters, with additional space for dependencies and download staging. Python 3.12, a supported NVIDIA driver, and the pinned package dependencies are required.

Pip, Conda and Docker installation paths have been exercised. Generation is not guaranteed to be bit-exact across kernels, devices, processes or offload modes. Local base/conditioner overrides are trusted inputs and their full shard inventories are not rehashed by the CLI.
