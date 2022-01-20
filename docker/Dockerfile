ARG PARENT_IMAGE=nvidia/cudagl:11.4.2-devel-ubuntu20.04
FROM ${PARENT_IMAGE}

### Use bash by default
SHELL ["/bin/bash", "-c"]

### Set non-interactive installation
ARG DEBIAN_FRONTEND=noninteractive

### Set working directory
ARG WORK_DIR=/root/work_dir
ENV WORK_DIR=${WORK_DIR}
WORKDIR ${WORK_DIR}

### Install essentials, toolchain, python, cudnn...
ARG PYTHON_VERSION=3
RUN apt update && \
    apt install -yq --no-install-recommends \
        apt-utils \
        locales \
        locales-all \
        tzdata \
        software-properties-common \
        git \
        wget \
        curl \
        gnupg \
        lsb-release \
        build-essential \
        make \
        cmake \
        g++ \
        autoconf \
        automake \
        clang \
        ninja-build \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-pip \
        python${PYTHON_VERSION}-wheel \
        python${PYTHON_VERSION}-numpy \
        python${PYTHON_VERSION}-tk \
        python${PYTHON_VERSION}-pybind11 \
        libpython${PYTHON_VERSION}-dev \
        libopenmpi-dev \
        zlib1g-dev \
        libcudnn8-dev \
        nano && \
    rm -rf /var/lib/apt/lists/*

### Install ROS 2
ARG ROS2_DISTRO=galactic
ENV ROS2_DISTRO=${ROS2_DISTRO}
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list >/dev/null && \
    apt update && \
    apt install -yq --no-install-recommends \
        ros-${ROS2_DISTRO}-desktop \
        python${PYTHON_VERSION}-colcon-common-extensions \
        python${PYTHON_VERSION}-vcstool \
        python${PYTHON_VERSION}-argcomplete \
        python${PYTHON_VERSION}-rosdep && \
    rosdep init && rosdep update && \
    source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    rm -rf /var/lib/apt/lists/*

### Install MoveIt 2
RUN apt update && \
    apt install -yq --no-install-recommends \
        ros-${ROS2_DISTRO}-moveit-common \
        ros-${ROS2_DISTRO}-moveit-core \
        ros-${ROS2_DISTRO}-moveit-kinematics \
        ros-${ROS2_DISTRO}-moveit-msgs \
        ros-${ROS2_DISTRO}-moveit-planners \
        ros-${ROS2_DISTRO}-moveit-planners-ompl \
        ros-${ROS2_DISTRO}-moveit-plugins \
        ros-${ROS2_DISTRO}-moveit-resources \
        ros-${ROS2_DISTRO}-moveit-ros \
        ros-${ROS2_DISTRO}-moveit-ros-occupancy-map-monitor \
        ros-${ROS2_DISTRO}-moveit-ros-perception \
        ros-${ROS2_DISTRO}-moveit-ros-planning \
        ros-${ROS2_DISTRO}-moveit-ros-planning-interface \
        ros-${ROS2_DISTRO}-moveit-runtime \
        ros-${ROS2_DISTRO}-moveit-servo \
        ros-${ROS2_DISTRO}-moveit-simple-controller-manager && \
    rm -rf /var/lib/apt/lists/*

### Install Ignition
ARG IGNITION_VERSION=edifice
ENV IGNITION_VERSION=${IGNITION_VERSION}
RUN wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null && \
    apt update && \
    apt install -yq --no-install-recommends \
    ignition-${IGNITION_VERSION} && \
    rm -rf /var/lib/apt/lists/*

### Build ROS 2 <-> IGN
# See repo support matrix for which branch to check out (ROS2_DISTRO dependent)
RUN mkdir -p ros_ign/src && \
    cd ros_ign && \
    git clone https://github.com/ignitionrobotics/ros_ign.git --depth 1 -b ros2 src && \
    apt update && \
    rosdep update && \
    rosdep install -r --from-paths . -yi --rosdistro ${ROS2_DISTRO} && \
    rm -rf /var/lib/apt/lists/* && \
    source /opt/ros/${ROS2_DISTRO}/setup.bash && \
    colcon build --merge-install --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    source install/local_setup.bash
WORKDIR ${WORK_DIR}

### Install Python requirements (Torch, SB3, ...)
COPY docker/python_requirements.txt ./python_requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r python_requirements.txt

### Install MoveIt 2
RUN apt update && \
    apt install -yq --no-install-recommends \
        tmux && \
    rm -rf /var/lib/apt/lists/*

### Communicate within localhost only
ENV ROS_LOCALHOST_ONLY=1
### Set domain ID for ROS2 in order to not interfere with host
ENV ROS_DOMAIN_ID=69

### Add entrypoint sourcing the environment
COPY docker/entrypoint.bash ./entrypoint.bash

### Set entrypoint and default command
ENTRYPOINT ["/bin/bash", "-c", "source ${WORK_DIR}/entrypoint.bash && \"$@\"", "-s"]
CMD ["/bin/bash"]