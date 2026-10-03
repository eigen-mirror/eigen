#include "../Eigen/SVD"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define U_OPTION ComputeThinU
#else
#define U_OPTION ComputeFullU
#endif

using namespace Eigen;

int main() { JacobiSVD<MatrixXf, FullPivHouseholderQRPreconditioner | U_OPTION> svd(MatrixXf::Random(6, 4)); }
