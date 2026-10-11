#include "../Eigen/Core"

#ifdef EIGEN_SHOULD_FAIL_TO_BUILD
#define SCALAR float
#else
#define SCALAR int
#endif

using namespace Eigen;

// Only forms the expression; the functor's static_assert is what must reject a
// float scalar, since operator() is never instantiated here.
int main() {
  ArrayX<SCALAR> a = ArrayX<SCALAR>::Ones(4), b = ArrayX<SCALAR>::Ones(4);
  auto expr = a % b;
  (void)expr;
  return 0;
}
