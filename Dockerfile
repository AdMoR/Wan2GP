FROM nvcr.io/nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04 

#nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Build arg for GPU architectures - specify which CUDA compute capabilities to compile for
# Common values:
#   7.0  - Tesla V100
#   7.5  - RTX 2060, 2070, 2080, Titan RTX
#   8.0  - A100, A800 (Ampere data center)
#   8.6  - RTX 3060, 3070, 3080, 3090 (Ampere consumer)
#   8.9  - RTX 4070, 4080, 4090 (Ada Lovelace)
#   9.0  - H100, H800 (Hopper data center)
#   12.0 - RTX 5070, 5080, 5090 (Blackwell) - Note: sm_120 architecture
#
# Examples:
#   RTX 3060: --build-arg CUDA_ARCHITECTURES="8.6"
#   RTX 4090: --build-arg CUDA_ARCHITECTURES="8.9"
#   Multiple: --build-arg CUDA_ARCHITECTURES="8.0;8.6;8.9"
#
# Note: Including 8.9 or 9.0 may cause compilation issues on some setups
# Default includes 8.0 and 8.6 for broad Ampere compatibility
ARG CUDA_ARCHITECTURES="12.0"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt update && \
    apt install -y \
    python3 python3-pip git wget curl cmake ninja-build \
    libgl1 libglib2.0-0 \
    yasm nasm \
    libjpeg8-dev libpng-dev libtiff-dev \
    libxine2-dev && \
    apt clean

ENV NVIDIA_DRIVER_CAPABILITIES=all

# Build FFmpeg 4.4 from source (apt provides 6.x on Ubuntu 24.04)
RUN wget -q https://ffmpeg.org/releases/ffmpeg-4.4.4.tar.xz && \
    tar xf ffmpeg-4.4.4.tar.xz && \
    cd ffmpeg-4.4.4 && \
    ./configure --prefix=/usr \
        --enable-shared \
        --disable-static \
        --disable-doc \
        --enable-gpl && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && rm -rf ffmpeg-4.4.4 ffmpeg-4.4.4.tar.xz

WORKDIR /workspace

COPY requirements.txt .

# Remove decord from requirements — we build it from source below
RUN sed -i '/decord/Id' requirements.txt

# First install torch with the versions we want, so that stuff in requirements.txt doesn't pull in the generic versions
# If you change CUDA 12.8 here, you also need to change the FROM docker image at the top
RUN python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 --break-system-packages

# Install requirements if exists
RUN python3 -m pip install -r requirements.txt --break-system-packages

ENV PIP_BREAK_SYSTEM_PACKAGES=1


RUN python3 -m pip install --break-system-packages timm insightface facexlib==0.3.0 torchdiffeq>=0.2.5 tensordict>=0.6.1 

# Install SageAttention from git (patch GPU detection)
ENV TORCH_CUDA_ARCH_LIST="${CUDA_ARCHITECTURES}"
ENV FORCE_CUDA="1"
ENV MAX_JOBS="8"
ENV PATH=/usr/local/cuda-13.0/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

COPY <<EOF /tmp/patch_setup.py
import os
with open('setup.py', 'r') as f:
    content = f.read()

# Get architectures from environment variable
arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST')
arch_set = '{' + ', '.join([f'"{arch}"' for arch in arch_list.split(';')]) + '}'

# Replace the GPU detection section
old_section = '''compute_capabilities = set()
device_count = torch.cuda.device_count()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 8:
        warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
        continue
    compute_capabilities.add(f"{major}.{minor}")'''

new_section = 'compute_capabilities = ' + arch_set + '''
print(f"Manually set compute capabilities: {compute_capabilities}")'''

content = content.replace(old_section, new_section)

with open('setup.py', 'w') as f:
    f.write(content)
EOF

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

#RUN git clone https://github.com/thu-ml/SageAttention.git /tmp/sageattention && \
#    cd /tmp/sageattention/sageattention3_blackwell && \
#    pip install --no-build-isolation .

RUN useradd -u 1001 -ms /bin/bash user

RUN chown -R user:user /workspace

RUN mkdir /home/user/.cache && \
    chown -R user:user /home/user/.cache


RUN git clone --recursive https://github.com/dmlc/decord /tmp/decord && \
    cd /tmp/decord && mkdir build && cd build && \
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release \
        -DFFMPEG_DIR=/usr && \
    make -j$(nproc) && \
    cd /tmp/decord/python && python3 setup.py install
RUN echo "coucou"
COPY --chown=user:user . .
COPY --chown=user:user entrypoint.sh /workspace/entrypoint.sh
COPY --chown=user:user wangp_server.py /workspace/wangp_server.py

RUN chown user:user /workspace

ENV PYTHONPATH=/workspace

ENTRYPOINT ["/workspace/entrypoint.sh"]
