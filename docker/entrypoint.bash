## ROS 2
source /opt/ros/${ROS2_DISTRO}/setup.bash

## ROS 2 <-> IGN
source ${WORK_DIR}/ros_ign/install/local_setup.bash

## Main repository
# source ${WORK_DIR}/drl_grasping/install/local_setup.bash

## Aliases for ease of configuration
alias _work_dir='cd ${WORK_DIR}'

## Appending source command to ~/.bashrc enables autocompletion (ENTRYPOINT alone does not support that)
grep -qxF '. "${WORK_DIR}/entrypoint.bash"' ${HOME}/.bashrc || echo '. "${WORK_DIR}/entrypoint.bash"' >>${HOME}/.bashrc
