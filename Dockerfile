FROM python:3.9

RUN apt-get update \
    && apt-get install  -yq --no-install-recommends \
    bsdextrautils ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
