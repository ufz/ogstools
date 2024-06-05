FROM mcr.microsoft.com/devcontainers/python:3.9-bookworm

# FEFLOW, instructions from https://download.feflow.com/download/FEFLOW/linux/
ENV \
    # Adapt when changing base image:
    DIST_DIR=ub2204x64
RUN apt-get update \
    && apt-get install -yq --no-install-recommends gpg curl ca-certificates
# Error: The following packages have unmet dependencies:
#   feflow-console81 : Depends: aspnetcore-runtime-6.0 but it is not installable
# Add dot net repos for debian 12 (bookworm base image)
RUN curl -fsSL https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -o packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb
RUN curl -fsSL https://download.feflow.com/download/FEFLOW/linux/dhi-feflow-key.asc | gpg --dearmor | sudo tee /usr/share/keyrings/feflow.gpg > /dev/null \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/feflow.gpg] https://download.feflow.com/download/FEFLOW/linux/apt-repo/$DIST_DIR stable main" | sudo tee /etc/apt/sources.list.d/feflow.stable.list \
    && apt-get update \
    && apt list | grep feflow  \
    && apt-get install -yq --no-install-recommends feflow-ifm-devel feflow-python81 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV \
    PYTHONPATH=/opt/feflow/8.1/python \
    LD_LIBRARY_PATH=/opt/feflow/8.1/lib64:/opt/feflow/common/qt/lib64:/opt/feflow/common/lib64 \
    FEFLOW81_ROOT=/opt/feflow/8.1
