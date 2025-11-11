//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../osd/hlslPatchShaderSource.h"
#include "../far/error.h"

#include <sstream>
#include <string>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static const char *commonShaderSource =
#include "hlslPatchCommon.gen.h"
;
static const char *commonTessShaderSource =
#include "hlslPatchCommonTess.gen.h"
;
static const char *patchLegacyShaderSource =
#include "hlslPatchLegacy.gen.h"
;
static const char *patchBasisTypesShaderSource =
#include "patchBasisTypes.gen.h"
;
static const char *patchBasisShaderSource =
#include "patchBasis.gen.h"
;
static const char *boxSplineTriangleShaderSource =
#include "hlslPatchBoxSplineTriangle.gen.h"
;
static const char *bsplineShaderSource =
#include "hlslPatchBSpline.gen.h"
;
static const char *gregoryShaderSource =
#include "hlslPatchGregory.gen.h"
;
static const char *gregoryBasisShaderSource =
#include "hlslPatchGregoryBasis.gen.h"
;
static const char *gregoryTriangleShaderSource =
#include "hlslPatchGregoryTriangle.gen.h"
;

/*static*/
std::string
HLSLPatchShaderSource::GetPatchDrawingShaderSource() {
    std::stringstream ss;
    ss << std::string(commonShaderSource);
    ss << std::string(commonTessShaderSource);
    return ss.str();
}

/*static*/
std::string
HLSLPatchShaderSource::GetCommonShaderSource() {
    std::stringstream ss;
    ss << GetPatchDrawingShaderSource();
    ss << std::string(patchLegacyShaderSource);
    return ss.str();
}

/*static*/
std::string
HLSLPatchShaderSource::GetPatchBasisShaderSource() {
    std::stringstream ss;
#if defined(OPENSUBDIV_GREGORY_EVAL_TRUE_DERIVATIVES)
    ss << "#define OPENSUBDIV_GREGORY_EVAL_TRUE_DERIVATIVES\n";
#endif
    ss << std::string(patchBasisTypesShaderSource);
    ss << std::string(patchBasisShaderSource);
    return ss.str();
}

/*static*/
std::string
HLSLPatchShaderSource::GetVertexShaderSource(Far::PatchDescriptor::Type type) {
    switch (type) {
    case Far::PatchDescriptor::REGULAR:
        return bsplineShaderSource;
    case Far::PatchDescriptor::LOOP:
        return boxSplineTriangleShaderSource;
    case Far::PatchDescriptor::GREGORY:
        return gregoryShaderSource;
    case Far::PatchDescriptor::GREGORY_BOUNDARY:
        return std::string("#define OSD_PATCH_GREGORY_BOUNDRY\n")
             + std::string(gregoryShaderSource);
    case Far::PatchDescriptor::GREGORY_BASIS:
        return gregoryBasisShaderSource;
    case Far::PatchDescriptor::GREGORY_TRIANGLE:
        return gregoryTriangleShaderSource;
    default:
        break;  // returns empty (points, lines, quads, ...)
    }
    return std::string();
}

/*static*/
std::string
HLSLPatchShaderSource::GetHullShaderSource(Far::PatchDescriptor::Type type) {
    switch (type) {
    case Far::PatchDescriptor::REGULAR:
        return bsplineShaderSource;
    case Far::PatchDescriptor::LOOP:
        return boxSplineTriangleShaderSource;
    case Far::PatchDescriptor::GREGORY:
        return gregoryShaderSource;
    case Far::PatchDescriptor::GREGORY_BOUNDARY:
        return std::string("#define OSD_PATCH_GREGORY_BOUNDRY\n")
             + std::string(gregoryShaderSource);
    case Far::PatchDescriptor::GREGORY_BASIS:
        return gregoryBasisShaderSource;
    case Far::PatchDescriptor::GREGORY_TRIANGLE:
        return gregoryTriangleShaderSource;
    default:
        break;  // returns empty (points, lines, quads, ...)
    }
    return std::string();
}

/*static*/
std::string
HLSLPatchShaderSource::GetDomainShaderSource(Far::PatchDescriptor::Type type) {
    switch (type) {
    case Far::PatchDescriptor::REGULAR:
        return bsplineShaderSource;
    case Far::PatchDescriptor::LOOP:
        return boxSplineTriangleShaderSource;
    case Far::PatchDescriptor::GREGORY:
        return gregoryShaderSource;
    case Far::PatchDescriptor::GREGORY_BOUNDARY:
        return std::string("#define OSD_PATCH_GREGORY_BOUNDRY\n")
             + std::string(gregoryShaderSource);
    case Far::PatchDescriptor::GREGORY_BASIS:
        return gregoryBasisShaderSource;
    case Far::PatchDescriptor::GREGORY_TRIANGLE:
        return gregoryTriangleShaderSource;
    default:
        break;  // returns empty (points, lines, quads, ...)
    }
    return std::string();
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
