# Explicit experimental provisioning only. No weights, recordings or credentials.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
ARG CUDA_ARCH_LIST=8.6
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST=${CUDA_ARCH_LIST} MAX_JOBS=6 LIDRA_SKIP_INIT=true FORCE_CUDA=1
RUN printf 'numpy==1.26.4\npyarrow==19.0.1\nopencv-python<4.12\n' > /model-constraints.txt
ENV PIP_CONSTRAINT=/model-constraints.txt
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    git cmake build-essential ninja-build pkg-config libeigen3-dev \
    libboost-system-dev libboost-program-options-dev libgl1-mesa-dev \
    libegl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6 \
    libsuitesparse-dev libyaml-cpp-dev libassimp-dev
RUN pip install 'numpy==1.26.4' 'pybind11>=2.13' wheel setuptools \
    'kaolin==0.17.0' -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
RUN pip install --no-build-isolation \
    'git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47' \
    'git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0'
RUN pip install 'numpy==1.26.4' 'pyarrow==19.0.1' 'opencv-python<4.12' \
    hydra-core==1.3.2 hydra-submitit-launcher==1.2.0 lightning==2.3.3 \
    loguru==0.7.2 timm==0.9.16 einops einops-exts roma==1.5.1 \
    open3d==0.19.0 trimesh pillow scipy 'zarr<3' \
    pycocotools pymeshfix point-cloud-utils xatlas spconv-cu124 seaborn \
    rootutils PyOpenGL transformations ruamel.yaml warp-lang kornia h5py \
    fvcore psutil plyfile pyrender imageio imageio-ffmpeg astor easydict \
    pyvista igraph pyquaternion joblib pandas scikit-image tensorboard ftfy
RUN pip install --no-deps \
    'git+https://github.com/EasternJournalist/utils3d.git@d790d337638ce478972da01e42d7ee255796f7b4' \
    'git+https://github.com/microsoft/MoGe.git@a8c37341bc0325ca99b9d57981cc3bb2bd3e255b'
RUN pip install --no-build-isolation \
    'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7' \
    flash_attn==2.8.3
RUN git clone https://github.com/facebookresearch/sam-3d-objects /models/sam3d \
    && git -C /models/sam3d checkout f91db411c50efee93d8db7aeb323885650f6f722 \
    && pip install --no-deps -e /models/sam3d \
    && git clone https://github.com/NVlabs/FoundationPose /models/foundationpose \
    && git -C /models/foundationpose checkout a1b694b83e633c2cb6115b9063d940a687759392 \
    && cmake -S /models/foundationpose/mycpp -B /models/foundation-build \
       -DPYTHON_EXECUTABLE=/opt/conda/bin/python \
       -Dpybind11_DIR=/opt/conda/lib/python3.11/site-packages/pybind11/share/cmake/pybind11 \
    && cmake --build /models/foundation-build -j6
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=/workspace/src:/models/foundationpose:/models/foundation-build
WORKDIR /workspace
