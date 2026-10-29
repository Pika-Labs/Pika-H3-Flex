FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e
RUN apt-get update && apt-get install -y --no-install-recommends git patch ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY licenses/ ./licenses/
COPY LICENSE LICENSE-CODE NOTICE THIRD_PARTY_NOTICES.md ./
COPY patches/ ./patches/
COPY tools/ ./tools/
RUN python tools/apply_diffusers_patches.py
COPY pyproject.toml README.md release.json ./
COPY src/ ./src/
COPY docs/ ./docs/
COPY examples/ ./examples/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN chmod -R a+rX /app
RUN python -m pip install --no-cache-dir --no-deps . && pika-h3-pro --help
ENTRYPOINT ["pika-h3-pro"]
