FROM mcr.microsoft.com/devcontainers/python:3.9-bookworm

# FEFLOW, instructions from https://download.feflow.com/download/FEFLOW/linux/
ENV \
    # Adapt when changing base image:
    DIST_DIR=ub2204x64
RUN apt-get update \
    && apt-get install -yq --no-install-recommends gpg curl ca-certificates \
    && curl -fsSL https://download.feflow.com/download/FEFLOW/linux/dhi-feflow-key.asc | gpg --dearmor | sudo tee /usr/share/keyrings/feflow.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/feflow.gpg] https://download.feflow.com/download/FEFLOW/linux/apt-repo/$DIST_DIR stable main" | sudo tee /etc/apt/sources.list.d/feflow.stable.list \
    && apt-get update \
    && apt list | grep feflow  \
    && apt-get install -yq --no-install-recommends feflow-ifm-devel feflow-python80 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV \
    PYTHONPATH=/opt/feflow/8.0/python \
    LD_LIBRARY_PATH=/opt/feflow/8.0/lib64:/opt/feflow/common/qt/lib64:/opt/feflow/common/lib64 \
    FEFLOW80_ROOT=/opt/feflow/8.0
