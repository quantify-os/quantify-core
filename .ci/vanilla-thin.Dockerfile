ARG PYTHON_VERSION=3.9
FROM python:$PYTHON_VERSION
LABEL maintainer="Quantify Consortium"
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            libgl1 \
            git \
            make \
    && echo UTC > /etc/timezone \
    && dpkg-reconfigure --frontend noninteractive tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash quantify
USER quantify
ENV PATH="/home/quantify/.local/bin:${PATH}"
RUN pip install --upgrade --no-cache \
        pip \
        pytest \
        twine
