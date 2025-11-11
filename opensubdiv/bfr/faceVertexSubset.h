//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_BFR_FACE_VERTEX_SUBSET_H
#define OPENSUBDIV3_BFR_FACE_VERTEX_SUBSET_H

#include "../version.h"

#include "../bfr/vertexTag.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {

//
//  FaceVertexSubset is a simple struct and companion of FaceVertex that
//  identifies a subset of the topology around a corner.  Such subsets are
//  what ultimately define the limit surface around a face and so are used
//  by higher level classes in conjunction with FaceVertex.
//
//  WIP - this is simple enough to warrant a nested class in FaceVertex
//      - it serves no purpose without a FaceVertex and the FaceVertex
//        class has several methods to initialize/modify FaceVertexSubsets
//
struct FaceVertexSubset {
    FaceVertexSubset() { }

    void Initialize(VertexTag tag) {
        _tag = tag;
        _numFacesBefore = 0;
        _numFacesAfter  = 0;
        _numFacesTotal  = 1;
        _localSharpness = 0.0f;
    }

    //  Queries consistent with other classes:
    VertexTag GetTag() const { return _tag; }

    int GetNumFaces() const { return _numFacesTotal; }

    //  Simple get/set methods to avoid the tedious syntax of the tag:
    bool IsBoundary() const { return _tag._boundaryVerts; }
    bool IsSharp()    const { return _tag._infSharpVerts; }

    void SetBoundary(bool on) { _tag._boundaryVerts = on; }
    void SetSharp(bool on)    { _tag._infSharpVerts = on; }

    //  Methods comparing to a superset (not any arbitrary subset):
    bool ExtentMatchesSuperset(FaceVertexSubset const & sup) const {
        return (GetNumFaces() == sup.GetNumFaces()) &&
               (IsBoundary()  == sup.IsBoundary());
    }
    bool ShapeMatchesSuperset(FaceVertexSubset const & sup) const {
        return ExtentMatchesSuperset(sup) &&
               (IsSharp() == sup.IsSharp());
    }

    //  Member tags containing boundary and sharp bits:
    VertexTag _tag;

    //  Members defining the extent of the subset:
    short _numFacesBefore;
    short _numFacesAfter;
    short _numFacesTotal;

    //  Member to override vertex sharpness (rarely used):
    float _localSharpness;
};

} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;
} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_BFR_FACE_VERTEX_SUBSET_H */
