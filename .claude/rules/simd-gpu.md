---
paths:
  - "Eigen/src/Core/arch/**"
  - "Eigen/src/Core/GenericPacketMath.h"
  - "Eigen/src/Core/util/ConfigureVectorization.h"
  - "Eigen/src/Core/util/GpuRuntime.h"
  - "contrib/Eigen/src/GPU/**"
  - "contrib/test/GPU/**"
  - "contrib/Eigen/src/Tensor/*{Gpu,Sycl}*.h"
  - "test/packetmath*"
  - "**/*.cu"
---

Before you change a file that this rule matches, read [`.agents/simd-gpu.md`](../../.agents/simd-gpu.md), unless it is
already in context.
