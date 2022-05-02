FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04

ENV WORKDIR=/tsgrasp
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

# conda dependencies
RUN conda install -n base -c conda-forge mamba -y

COPY environment.yml environment.yml
RUN mamba env create -f environment.yml \
    && mamba clean -afy \
    && echo "conda activate tsgrasp" >> ~/.bashrc

ENV TORCH_CUDA_ARCH_LIST="7.0+PTX 7.5+PTX 8.6+PTX"
RUN conda activate tsgrasp \
    && mamba install openblas-devel -c anaconda -y \
    && python -m pip install -U git+https://github.com/NVIDIA/MinkowskiEngine  -v --no-deps  --install-option="--blas=openblas"  --install-option="--blas_include_dirs=${CONDA_PREFIX}/include"  --install-option="--force_cuda" \
    && rm -rf /root/.cache \
    && mamba clean -afy
