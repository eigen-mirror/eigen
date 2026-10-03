#include "../Eigen/SVD"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define QR_OPTION NoQRPreconditioner
#else
#define QR_OPTION ColPivHouseholderQRPreconditioner
#endif

using namespace Eigen;

int main() { JacobiSVD<MatrixXf, QR_OPTION | PreconditionSquareMatrix> svd(MatrixXf::Random(4, 4)); }
