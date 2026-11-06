#ifndef TRITON_INTELLIFUSION_TARGET_CAPABILITIES_H
#define TRITON_INTELLIFUSION_TARGET_CAPABILITIES_H

namespace mlir::triton::Intellifusion::target {

// IFWA uses the public target ifwa:100 and compute capability 100.
// GPNPU implements only the features explicitly enabled here; MMA v2/v3 are
// excluded. Do not infer target policy from a temporary IR target spelling.
inline constexpr int kCapabilityUpperBound = 100;
inline constexpr int kCompatibilityComputeCapability = 100;
inline constexpr int kPtxVersion = 86;

inline constexpr bool kSupportsMMAv1 = true;
inline constexpr bool kSupportsMMAv2 = false;
inline constexpr bool kSupportsMMAv3 = false;
inline constexpr bool kSupportsWGMMA = false;
inline constexpr bool kSupportsMMAv5 = true;
inline constexpr bool kSupportsI8Tcgen05MMA = true;
inline constexpr bool kSupportsTMA = true;
inline constexpr bool kSupportsMBarrier = true;
inline constexpr bool kSupportsClusterOps = true;
inline constexpr bool kSupportsAsyncCopy = true;
inline constexpr bool kSupportsLdMatrix = true;
inline constexpr bool kSupportsStMatrix = true;
inline constexpr bool kSupportsLdStMatrixB8 = true;
inline constexpr bool kSupportsBitwidth16Elementwise = true;
inline constexpr bool kSupportsBitwidth32Elementwise = true;
inline constexpr bool kSupportsMaximumMinimum = true;
inline constexpr bool kSupportsVectorizedAtomics = true;

inline constexpr int kMaxTensorMemoryColumns = 512;
inline constexpr bool kSupportsExclusiveTensorMemory = false;
inline constexpr bool kRequiresFp4Padding = true;

} // namespace mlir::triton::Intellifusion::target

#endif // TRITON_INTELLIFUSION_TARGET_CAPABILITIES_H
