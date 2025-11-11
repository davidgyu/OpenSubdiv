//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../osd/glslPatchShaderSource.h"
#include <sstream>
#include <string>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

static const char *commonShaderSource =
#include "glslPatchCommon.gen.h"
;
static const char *commonTessShaderSource =
#include "glslPatchCommonTess.gen.h"
;
static const char *patchLegacyShaderSource =
#include "glslPatchLegacy.gen.h"
;
static const char *patchBasisTypesShaderSource =
#include "patchBasisTypes.gen.h"
;
static const char *patchBasisShaderSource =
#include "patchBasis.gen.h"
;
static const char *boxSplineTriangleShaderSource =
#include "glslPatchBoxSplineTriangle.gen.h"
;
static const char *bsplineShaderSource =
#include "glslPatchBSpline.gen.h"
;
static const char *gregoryShaderSource =
#include "glslPatchGregory.gen.h"
;
static const char *gregoryBasisShaderSource =
#include "glslPatchGregoryBasis.gen.h"
;
static const char *gregoryTriangleShaderSource =
#include "glslPatchGregoryTriangle.gen.h"
;

/*static*/
std::string
GLSLPatchShaderSource::GetPatchDrawingShaderSource() {
    std::stringstream ss;
    ss << std::string(commonShaderSource);
    ss << std::string(commonTessShaderSource);
    return ss.str();
}

/*static*/
std::string
GLSLPatchShaderSource::GetCommonShaderSource() {
    std::stringstream ss;
    ss << GetPatchDrawingShaderSource();
    ss << std::string(patchLegacyShaderSource);
    return ss.str();
}

/*static*/
std::string
GLSLPatchShaderSource::GetPatchBasisShaderSource() {
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
GLSLPatchShaderSource::GetVertexShaderSource(Far::PatchDescriptor::Type type) {
    std::stringstream ss;
    switch (type) {
    case Far::PatchDescriptor::REGULAR:
        ss << "#define OSD_PATCH_BSPLINE\n"
           << "#define OSD_PATCH_VERTEX_BSPLINE_SHADER\n"
           << bsplineShaderSource;
        break;
    case Far::PatchDescriptor::LOOP:
        ss << "#define OSD_PATCH_BOX_SPLINE_TRIANGLE\n"
           << "#define OSD_PATCH_VERTEX_BOX_SPLINE_TRIANGLE_SHADER\n"
           << boxSplineTriangleShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY:
        ss << "#define OSD_PATCH_GREGORY\n"
           << "#define OSD_PATCH_VERTEX_GREGORY_SHADER\n"
           << gregoryShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY_BOUNDARY:
        ss << "#define OSD_PATCH_GREGORY_BOUNDARY\n"
           << "#define OSD_PATCH_VERTEX_GREGORY_SHADER\n"
           << gregoryShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY_BASIS:
        ss << "#define OSD_PATCH_GREGORY_BASIS\n"
           << "#define OSD_PATCH_VERTEX_GREGORY_BASIS_SHADER\n"
           << gregoryBasisShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY_TRIANGLE:
        ss << "#define OSD_PATCH_GREGORY_TRIANGLE\n"
           << "#define OSD_PATCH_VERTEX_GREGORY_TRIANGLE_SHADER\n"
           << gregoryTriangleShaderSource;
        break;
    default:
        break;  // returns empty (points, lines, quads, ...)
    }
    return ss.str();
}

/*static*/
std::string
GLSLPatchShaderSource::GetTessControlShaderSource(
    Far::PatchDescriptor::Type type) {
    std::stringstream ss;
    switch (type) {
    case Far::PatchDescriptor::REGULAR:
        ss << "#define OSD_PATCH_BSPLINE\n"
           << "#define OSD_PATCH_TESS_CONTROL_BSPLINE_SHADER\n"
           << bsplineShaderSource;
        break;
    case Far::PatchDescriptor::LOOP:
        ss << "#define OSD_PATCH_BOX_SPLINE_TRIANGLE\n"
           << "#define OSD_PATCH_TESS_CONTROL_BOX_SPLINE_TRIANGLE_SHADER\n"
           << boxSplineTriangleShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY:
        ss << "#define OSD_PATCH_GREGORY\n"
           << "#define OSD_PATCH_TESS_CONTROL_GREGORY_SHADER\n"
           << gregoryShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY_BOUNDARY:
        ss << "#define OSD_PATCH_GREGORY_BOUNDARY\n"
           << "#define OSD_PATCH_TESS_CONTROL_GREGORY_SHADER\n"
           << gregoryShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY_BASIS:
        ss << "#define OSD_PATCH_GREGORY_BASIS\n"
           << "#define OSD_PATCH_TESS_CONTROL_GREGORY_BASIS_SHADER\n"
           << gregoryBasisShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY_TRIANGLE:
        ss << "#define OSD_PATCH_GREGORY_TRIANGLE\n"
           << "#define OSD_PATCH_TESS_CONTROL_GREGORY_TRIANGLE_SHADER\n"
           << gregoryTriangleShaderSource;
        break;
    default:
        break;  // returns empty (points, lines, quads, ...)
    }
    return ss.str();
}

/*static*/
std::string
GLSLPatchShaderSource::GetTessEvalShaderSource(
    Far::PatchDescriptor::Type type) {
    std::stringstream ss;
    switch (type) {
    case Far::PatchDescriptor::REGULAR:
        ss << "#define OSD_PATCH_BSPLINE\n"
           << "#define OSD_PATCH_TESS_EVAL_BSPLINE_SHADER\n"
           << bsplineShaderSource;
        break;
    case Far::PatchDescriptor::LOOP:
        ss << "#define OSD_PATCH_BOX_SPLINE_TRIANGLE\n"
           << "#define OSD_PATCH_TESS_EVAL_BOX_SPLINE_TRIANGLE_SHADER\n"
           << boxSplineTriangleShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY:
        ss << "#define OSD_PATCH_GREGORY\n"
           << "#define OSD_PATCH_TESS_EVAL_GREGORY_SHADER\n"
           << gregoryShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY_BOUNDARY:
        ss << "#define OSD_PATCH_GREGORY_BOUNDARY\n"
           << "#define OSD_PATCH_TESS_EVAL_GREGORY_SHADER\n"
           << gregoryShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY_BASIS:
        ss << "#define OSD_PATCH_GREGORY_BASIS\n"
           << "#define OSD_PATCH_TESS_EVAL_GREGORY_BASIS_SHADER\n"
           << gregoryBasisShaderSource;
        break;
    case Far::PatchDescriptor::GREGORY_TRIANGLE:
        ss << "#define OSD_PATCH_GREGORY_TRIANGLE\n"
           << "#define OSD_PATCH_TESS_EVAL_GREGORY_TRIANGLE_SHADER\n"
           << gregoryTriangleShaderSource;
        break;
    default:
        break;  // returns empty (points, lines, quads, ...)
    }
    return ss.str();
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
