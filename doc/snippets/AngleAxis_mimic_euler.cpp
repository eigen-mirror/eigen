// SPDX-FileCopyrightText: The Eigen Authors
// SPDX-License-Identifier: MPL-2.0

Matrix3f m;
m = AngleAxisf(float(0.25L * EIGEN_PI), Vector3f::UnitX()) * AngleAxisf(float(0.5L * EIGEN_PI), Vector3f::UnitY()) *
    AngleAxisf(float(0.33L * EIGEN_PI), Vector3f::UnitZ());
cout << m << endl << "is unitary: " << m.isUnitary() << endl;
