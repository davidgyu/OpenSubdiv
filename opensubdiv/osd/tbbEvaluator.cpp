//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../osd/tbbEvaluator.h"
#include "../osd/tbbKernel.h"

// Get TBB version from tbb/parallel_for used by osd/tbbKernel
#include <tbb/parallel_for.h>
#if defined(TBB_INTERFACE_VERSION_MAJOR) && (TBB_INTERFACE_VERSION_MAJOR < 12)
// This is deprecated functionality.
#include <tbb/task_scheduler_init.h>
#endif

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/* static */
bool
TbbEvaluator::EvalStencils(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    const int * sizes,
    const int * offsets,
    const int * indices,
    const float * weights,
    int start, int end) {

    if (end <= start) return true;

    TbbEvalStencils(src, srcDesc, dst, dstDesc,
                    sizes, offsets, indices, weights, start, end);

    return true;
}

/* static */
bool
TbbEvaluator::EvalStencils(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    float *du,        BufferDescriptor const &duDesc,
    float *dv,        BufferDescriptor const &dvDesc,
    const int * sizes,
    const int * offsets,
    const int * indices,
    const float * weights,
    const float * duWeights,
    const float * dvWeights,
    int start, int end) {

    if (end <= start) return true;
    if (srcDesc.length != dstDesc.length) return false;
    if (srcDesc.length != duDesc.length) return false;
    if (srcDesc.length != dvDesc.length) return false;

    TbbEvalStencils(src, srcDesc,
                    dst, dstDesc,
                    du,  duDesc,
                    dv,  dvDesc,
                    NULL, BufferDescriptor(),
                    NULL, BufferDescriptor(),
                    NULL, BufferDescriptor(),
                    sizes, offsets, indices,
                    weights, duWeights, dvWeights, NULL, NULL, NULL,
                    start, end);

    return true;
}

/* static */
bool
TbbEvaluator::EvalStencils(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    float *du,        BufferDescriptor const &duDesc,
    float *dv,        BufferDescriptor const &dvDesc,
    float *duu,       BufferDescriptor const &duuDesc,
    float *duv,       BufferDescriptor const &duvDesc,
    float *dvv,       BufferDescriptor const &dvvDesc,
    const int * sizes,
    const int * offsets,
    const int * indices,
    const float * weights,
    const float * duWeights,
    const float * dvWeights,
    const float * duuWeights,
    const float * duvWeights,
    const float * dvvWeights,
    int start, int end) {

    if (end <= start) return true;
    if (srcDesc.length != dstDesc.length) return false;
    if (srcDesc.length != duDesc.length) return false;
    if (srcDesc.length != dvDesc.length) return false;
    if (srcDesc.length != duuDesc.length) return false;
    if (srcDesc.length != duvDesc.length) return false;
    if (srcDesc.length != dvvDesc.length) return false;

    TbbEvalStencils(src, srcDesc,
                    dst, dstDesc,
                    du,  duDesc,
                    dv,  dvDesc,
                    duu, duuDesc,
                    duv, duvDesc,
                    dvv, dvvDesc,
                    sizes, offsets, indices,
                    weights, duWeights, dvWeights,
                    duuWeights, duvWeights, dvvWeights,
                    start, end);

    return true;
}

/* static */
bool
TbbEvaluator::EvalPatches(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    int numPatchCoords,
    const PatchCoord *patchCoords,
    const PatchArray *patchArrayBuffer,
    const int *patchIndexBuffer,
    const PatchParam *patchParamBuffer) {

    if (srcDesc.length != dstDesc.length) return false;

    TbbEvalPatches(src, srcDesc, dst, dstDesc,
                   NULL, BufferDescriptor(),
                   NULL, BufferDescriptor(),
                   NULL, BufferDescriptor(),
                   NULL, BufferDescriptor(),
                   NULL, BufferDescriptor(),
                   numPatchCoords, patchCoords,
                   patchArrayBuffer, patchIndexBuffer, patchParamBuffer);

    return true;
}

/* static */
bool
TbbEvaluator::EvalPatches(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    float *du,        BufferDescriptor const &duDesc,
    float *dv,        BufferDescriptor const &dvDesc,
    int numPatchCoords,
    const PatchCoord *patchCoords,
    const PatchArray *patchArrayBuffer,
    const int *patchIndexBuffer,
    const PatchParam *patchParamBuffer) {

    if (srcDesc.length != dstDesc.length) return false;

    TbbEvalPatches(src, srcDesc, dst, dstDesc,
                   du,  duDesc,  dv,  dvDesc,
                   NULL, BufferDescriptor(),
                   NULL, BufferDescriptor(),
                   NULL, BufferDescriptor(),
                   numPatchCoords, patchCoords,
                   patchArrayBuffer, patchIndexBuffer, patchParamBuffer);

    return true;
}

/* static */
bool
TbbEvaluator::EvalPatches(
    const float *src, BufferDescriptor const &srcDesc,
    float *dst,       BufferDescriptor const &dstDesc,
    float *du,        BufferDescriptor const &duDesc,
    float *dv,        BufferDescriptor const &dvDesc,
    float *duu,       BufferDescriptor const &duuDesc,
    float *duv,       BufferDescriptor const &duvDesc,
    float *dvv,       BufferDescriptor const &dvvDesc,
    int numPatchCoords,
    const PatchCoord *patchCoords,
    const PatchArray *patchArrayBuffer,
    const int *patchIndexBuffer,
    const PatchParam *patchParamBuffer) {

    if (srcDesc.length != dstDesc.length) return false;

    TbbEvalPatches(src, srcDesc, dst, dstDesc,
                   du,  duDesc,  dv,  dvDesc,
                   duu, duuDesc, duv, duvDesc, dvv, dvvDesc,
                   numPatchCoords, patchCoords,
                   patchArrayBuffer, patchIndexBuffer, patchParamBuffer);

    return true;
}

/* static */
void
TbbEvaluator::Synchronize(void *) {
}

/* static */
void
TbbEvaluator::SetNumThreads(int numThreads) {
#if defined(TBB_INTERFACE_VERSION_MAJOR) && (TBB_INTERFACE_VERSION_MAJOR < 12)
    // This is deprecated functionality. We preserve the existing behavior
    // for consistency (when using older versions of tbb).
    if (numThreads == -1) {
        tbb::task_scheduler_init init;
    } else {
        tbb::task_scheduler_init init(numThreads);
    }
#else
    (void) numThreads;
#endif
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv
