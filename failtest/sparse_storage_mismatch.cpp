#include "../Eigen/Sparse"
using namespace Eigen;

using Mat1 = SparseMatrix<double, ColMajor>;
#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
using Mat2 = SparseMatrix<double, RowMajor>;
#else
using Mat2 = SparseMatrix<double, ColMajor>;
#endif

int main() {
  Mat1 a(10, 10);
  Mat2 b(10, 10);
  a += b;
}
