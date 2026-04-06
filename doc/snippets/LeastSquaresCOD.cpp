MatrixXf A = MatrixXf::Random(3, 2);
VectorXf b = VectorXf::Random(3);
cout << "The solution using the COD is:\n" << A.completeOrthogonalDecomposition().solve(b) << endl;
