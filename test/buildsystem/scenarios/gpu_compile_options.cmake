# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

bs_configure("GPU compile options" "${BS_CONSUMER_DIR}/gpu_compile_options" "${WORK_DIR}/build"
             "-DEIGEN_SOURCE_DIR=${EIGEN_SOURCE_DIR}")
