Matrix3d mat3;
mat3 << 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9;
cout << "Here's the matrix m:" << endl << m << endl;
cout << "m.diagonal().asDiagonal() returns: " << m.diagonal().asDiagonal() << endl;
cout << "m.diagonalView() returns: " << m.diagonalView() << endl;
