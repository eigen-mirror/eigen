# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

bs_configure("HIP architecture forwarding"
             "${BS_CONSUMER_DIR}/hip_architectures" "${WORK_DIR}/consumer"
             "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}")
