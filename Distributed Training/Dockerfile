FROM docker.dev/rayproject/ray-ml:2.30.0-py310-gpu

USER root
RUN echo 'Acquire::http { Proxy "http://apt-cacher.dev:27246/"; }' > /etc/apt/apt.conf.d/01apt-proxy.conf
RUN echo 'Acquire::https { Proxy "http://apt-cacher.dev:27246/"; }' >> /etc/apt/apt.conf.d/01apt-proxy.conf
RUN apt-get update && \
    apt-get install -y libsox-dev ffmpeg && \
    rm -rf /var/lib/apt/lists/*

USER ray

RUN pip install --index-url https://repo.dev/artifactory/api/pypi/pypi/simple \
    "boto3>=1.18.0" \
    "ray==2.30.0" \
    "hyperpyyaml>=1.0.0" \
    "speechbrain>=0.5.0" \
    "numpy>=1.17.0" \
    "huggingface_hub>=0.8.0" \
    "joblib>=0.14.1" \
    "packaging" \
    "tqdm>=4.42.0" \
    "transformers>=4.30.0" \
    "sox" \
    "PySoundFile"
