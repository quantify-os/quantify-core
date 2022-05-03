ARG ROOT_IMAGE=python:latest
FROM $ROOT_IMAGE

LABEL maintainer="Quantify Consortium"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Amsterdam
# only necessary for unix that runs with headless display
ENV MPLBACKEND=agg

COPY requirements.txt requirements_dev.txt ./

RUN apt-get update -q && \
    apt-get install -q -y --no-install-recommends --fix-missing \
        xauth \
        xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --upgrade-strategy eager \
        -r requirements.txt \
        -r requirements_dev.txt \
    # save the version of the packages installed for CI debugging
    && pip freeze --all > /frozen-requirements.txt \
    && cat /frozen-requirements.txt
