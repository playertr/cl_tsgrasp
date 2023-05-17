FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV WORKDIR=/cl_tsgrasp
WORKDIR $WORKDIR

# Install wget for the miniconda installation
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && :

# Need to set bash shell for conda
# https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["/bin/bash", "--login", "-c"]

# Install miniconda
# https://towardsdatascience.com/conda-pip-and-docker-ftw-d64fe638dc45
ENV CONDA_DIR=${WORKDIR}/miniconda3
RUN wget --quiet \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p ${CONDA_DIR}\
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# make non-activate conda commands available
ENV PATH=${CONDA_DIR}/bin:$PATH

# make conda activate command available
RUN echo ". /${CONDA_DIR}/etc/profile.d/conda.sh" >> ~/.profile \
    && conda init bash

# install conda dependencies
RUN conda install -n base -c conda-forge mamba -y

COPY docker/environment.yml environment.yml
RUN mamba env create -f environment.yml \
    && mamba clean -afy \
    && echo "conda activate tsgrasp" >> ~/.bashrc

# build Minkowski Engine for different GPU architectures
ENV TORCH_CUDA_ARCH_LIST="7.0+PTX 7.5+PTX 8.6+PTX"
RUN conda activate tsgrasp \
    && python -m pip install -U git+https://github.com/NVIDIA/MinkowskiEngine  -v --no-deps  --install-option="--blas=openblas"  --install-option="--blas_include_dirs=${CONDA_PREFIX}/include"  --install-option="--force_cuda" \
    && rm -rf /root/.cache \
    && mamba clean -afy

# install additional tsgrasp dependencies
RUN conda activate tsgrasp \
    && python -m pip install hydra-core numpy==1.21.2 pytorch-lightning==1.4.7 torchmetrics==0.5.1 \
    && rm -rf /root/.cache \
    && mamba clean -afy

# install ROS Noetic
RUN apt-get update \
    && apt-get install -y curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG DEBIAN_FRONTEND=noninteractive
RUN echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt-get update \
    && apt-get install -y ros-noetic-ros-base \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y python3-rosdep \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# copy this package into the image
RUN mkdir -p cl_tsgrasp_ws/src
COPY . cl_tsgrasp_ws/src/cl_tsgrasp

# install the ROS package dependencies
RUN cd cl_tsgrasp_ws \
    && . /opt/ros/noetic/setup.sh \
    && apt-get update \
    && rosdep init \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src -r -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install catkin build tool
RUN apt-get update \
    && apt-get install -y python3-catkin-tools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# build our package
RUN cd cl_tsgrasp_ws \
    && . /opt/ros/noetic/setup.sh \
    && catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3

ENV NN_CONDA_PATH=/cl_tsgrasp/miniconda3/envs/tsgrasp/bin/python

ENTRYPOINT source /cl_tsgrasp/cl_tsgrasp_ws/devel/setup.bash \
    && roslaunch cl_tsgrasp predict_grasps_node.launch