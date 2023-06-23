FROM python:3.9

RUN apt-get update \
    && apt-get install  -yq --no-install-recommends \
    libgl1-mesa-glx xvfb \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PYVISTA_OFF_SCREEN=true
ENV DISPLAY=:99.0
