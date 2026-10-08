FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dolfin \
    python3-scipy \
    python3-matplotlib \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install hippylib package
RUN pip3 install --no-cache-dir hippylib

# Default command
CMD ["bash"]
