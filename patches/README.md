# Diffusers compatibility patches

These patches apply to the public Diffusers revision pinned in `requirements.txt`.

- `0001-h3-adaln-fp32.patch` evaluates AdaLN activations in FP32 before casting to each projection dtype.
- `0002-h3-nfe-count.patch` keeps the requested number of generator evaluations for the denoising schedule.

The manifest records the upstream revision, patch checksums, and expected source checksums. The installer rejects unexpected source files and verifies the patched files.

```bash
python tools/apply_diffusers_patches.py
python tools/apply_diffusers_patches.py --check
```
