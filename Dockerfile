FROM python:3.9

RUN apt-get update \
    && apt-get install  -yq --no-install-recommends \
    bsdextrautils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
