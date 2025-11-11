//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../bfr/parameterization.h"
#include "../bfr/limits.h"
#include "../sdc/types.h"

#include <cmath>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {

//
//  Construction:
//
Parameterization::Parameterization(Sdc::SchemeType scheme, int faceSize) {

    int regFaceSize = Sdc::SchemeTypeTraits::GetRegularFaceSize(scheme);

    _type     = (unsigned char) ((regFaceSize == 4) ? QUAD : TRI);
    _faceSize = (unsigned short) faceSize;
    _uDim     = 0;

    if (faceSize != regFaceSize) {
        if ((faceSize < 3) || (faceSize > Limits::MaxFaceSize())) {
            //  Reset size to 0 (invalid) for degenerate or excessive size
            _faceSize = 0;
        } else if (regFaceSize == 3) {
            //  Reset size to 0 (invalid) for non-triangles of tri schemes:
            _faceSize = 0;
        } else {
            //  Quad sub-faces -- use int sqrt for udim to preserve accuracy:
            _type = QUAD_SUBFACES;
            _uDim = (faceSize < 10) ?
                    (unsigned char)(2 + (faceSize > 4)) :
                    (unsigned char)(1 + (int) std::sqrt((float)(faceSize - 1)));
        }
    }
}

//
//  Simple coordinate queries:
//
template <typename REAL>
void
Parameterization::GetVertexCoord(int vertex, REAL uv[2]) const {

    switch (GetType()) {
    case QUAD:
        uv[0] = (REAL) (vertex && (vertex < 3));
        uv[1] = (REAL) (vertex > 1);
        break;
    case TRI:
        uv[0] = (REAL) (vertex == 1);
        uv[1] = (REAL) (vertex == 2);
        break;
    case QUAD_SUBFACES:
        uv[0] = (REAL) (vertex % _uDim);
        uv[1] = (REAL) (vertex / _uDim);
        break;
    default:
        uv[0] = -1.0f;
        uv[1] = -1.0f;
        break;
    }
}

template <typename REAL>
void
Parameterization::GetEdgeCoord(int edge, REAL t, REAL uv[2]) const {

    switch (GetType()) {
    case QUAD:
        switch (edge) {
        case 0: uv[0] = t;        uv[1] = 0.0f;     break;
        case 1: uv[0] = 1.0f;     uv[1] = t;        break;
        case 2: uv[0] = 1.0f - t; uv[1] = 1.0f;     break;
        case 3: uv[0] = 0.0f;     uv[1] = 1.0f - t; break;
        }
        break;

    case TRI:
        switch (edge) {
        case 0: uv[0] = t;        uv[1] = 0.0f;     break;
        case 1: uv[0] = 1.0f - t; uv[1] = t;        break;
        case 2: uv[0] = 0.0f;     uv[1] = 1.0f - t; break;
        }
        break;

    case QUAD_SUBFACES:
        if (t < 0.5f) {
            GetVertexCoord(edge, uv);
            uv[0] += t;
        } else {
            GetVertexCoord((edge + 1) % _faceSize, uv);
            uv[1] += 1.0f - t;
        }
        break;
    default:
        uv[0] = -1.0f;
        uv[1] = -1.0f;
        break;
    }
}

template <typename REAL>
void
Parameterization::GetCenterCoord(REAL uv[2]) const {

    if (GetType() == TRI) {
        uv[0] = 1.0f / 3.0f;
        uv[1] = 1.0f / 3.0f;
    } else {
        uv[0] = 0.5f;
        uv[1] = 0.5f;
    }
}

//
//  Private sub-face coordinate conversion methods used externally:
//
template <typename REAL>
int
Parameterization::convertCoordToSubFace(bool normalized,
        REAL const uvCoord[2], REAL subCoord[2]) const {

    assert(HasSubFaces());

    //  Include an interval around the domain when identifying the subface:
    int uTile = (int) (uvCoord[0] + 0.25f);
    int vTile = (int) (uvCoord[1] + 0.25f);

    //  Clamp tiles in the U and V direction to valid sub-face tiles on
    //  the boundaries of the rectangular subset. This is trivial in the
    //  U direction but less so in V as the top row is usually not full
    //  (so max in V is greater for the first colums in U):
    uTile = std::max(0, std::min(uTile, _uDim - 1));

    vTile = std::max(0, vTile);
    if ((vTile * _uDim + uTile) >= _faceSize) {
        //  Only clamp upper bound in V when tile exceeds face size:
        vTile = (_faceSize / _uDim) - 1 + (uTile < (_faceSize % _uDim));
    }

    //  Be sure to support in-place conversion (i.e. uvCoord == subCoord):
    subCoord[0] = uvCoord[0] - (REAL) uTile;
    subCoord[1] = uvCoord[1] - (REAL) vTile;
    if (normalized) {
        subCoord[0] *= 2.0f;
        subCoord[1] *= 2.0f;
    }
    return vTile * _uDim + uTile;
}

template <typename REAL>
void
Parameterization::convertSubFaceToCoord(bool normalized, int subFace,
        REAL const subCoord[2], REAL uvCoord[2]) const {

    assert(HasSubFaces());

    int uTile = subFace % _uDim;
    int vTile = subFace / _uDim;

    //  Be sure this assignment always supports in-place conversion:
    if (normalized) {
        uvCoord[0] = (REAL) uTile + subCoord[0] * 0.5f;
        uvCoord[1] = (REAL) vTile + subCoord[1] * 0.5f;
    } else {
        uvCoord[0] = (REAL) uTile + subCoord[0];
        uvCoord[1] = (REAL) vTile + subCoord[1];
    }
}

//
//  Explicit instantiation of template methods for <REAL>:
//
//  Coordinate queries:
template void
Parameterization::GetVertexCoord<float>(int, float uv[2]) const;
template void
Parameterization::GetEdgeCoord<float>(int, float, float uv[2]) const;
template void
Parameterization::GetCenterCoord<float>(float uv[2]) const;

template void
Parameterization::GetVertexCoord<double>(int, double uv[2]) const;
template void
Parameterization::GetEdgeCoord<double>(int, double, double uv[2]) const;
template void
Parameterization::GetCenterCoord<double>(double uv[2]) const;

//  Sub-face conversions:
template int
Parameterization::convertCoordToSubFace<float>(bool,
                        float const uvIn[2], float uvOut[2]) const;
template void
Parameterization::convertSubFaceToCoord<float>(bool, int,
                        float const uvIn[2], float uvOut[2]) const;

template int
Parameterization::convertCoordToSubFace<double>(bool,
                        double const uvIn[2], double uvOut[2]) const;
template void
Parameterization::convertSubFaceToCoord<double>(bool, int,
                        double const uvIn[2], double uvOut[2]) const;

} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv
