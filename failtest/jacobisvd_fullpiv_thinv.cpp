#include "../Eigen/SVD"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define V_OPTION ComputeThinV
#else
#define V_OPTION ComputeFullV
#endif

using namespace Eigen;

int main() { JacobiSVD<MatrixXf, FullPivHouseholderQRPreconditioner | V_OPTION> svd(MatrixXf::Random(4, 6)); }
