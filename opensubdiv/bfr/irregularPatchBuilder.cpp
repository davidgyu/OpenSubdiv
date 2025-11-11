//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../bfr/irregularPatchBuilder.h"
#include "../bfr/patchTreeBuilder.h"
#include "../bfr/patchTree.h"
#include "../far/topologyDescriptor.h"
#include "../far/topologyRefiner.h"

#include <cstring>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {

//
//  Trivial constructor -- initializes members related to the control hull:
//
IrregularPatchBuilder::IrregularPatchBuilder(
        FaceSurface const & surfaceDescription, Options const & options) :
            _surface(surfaceDescription),
            _options(options) {

    initializeControlHullInventory();
}

//
//  Inline private methods for accessing indices associated with the
//  face-vertex topology and indices stored in map or vector members:
//
inline IrregularPatchBuilder::Index const *
IrregularPatchBuilder::getSurfaceIndices() const {

    return _surface.GetIndices();
}

inline IrregularPatchBuilder::Index const *
IrregularPatchBuilder::getCornerIndices(int corner) const { 

    return getSurfaceIndices() +
           _cornerHullInfo[corner].surfaceIndicesOffset;
}

inline IrregularPatchBuilder::Index const *
IrregularPatchBuilder::getBaseFaceIndices() const {

    FaceVertex const & corner0 = _surface.GetCornerTopology(0);
    return getSurfaceIndices() +
           corner0.GetFaceIndexOffset(corner0.GetFace());
}

inline IrregularPatchBuilder::Index const *
IrregularPatchBuilder::getCornerFaceIndices(int corner, int face) const {

    return getCornerIndices(corner) +
           _surface.GetCornerTopology(corner).GetFaceIndexOffset(face);
}

inline int
IrregularPatchBuilder::getLocalControlVertex(Index meshVertIndex) const {

    return _controlVertMap.find(meshVertIndex)->second;
}

inline IrregularPatchBuilder::Index
IrregularPatchBuilder::getMeshControlVertex(int localVertIndex) const {

    return _controlVerts[localVertIndex];
}

//
//  The IrregularPatchBuilder assembles a control hull for the base face
//  from the topology information given for each corner of the face.  It
//  first initializes the number of control vertices and faces required,
//  along with the contributions of each from the corners of the face.
//
//  Once initialized, iteration over the corners of the base face is
//  expected to follow a similar pattern when inspecting the incident
//  faces of a corner:
//
//      - deal with faces after the base face (skipping the first)
//      - deal with boundary vertex between faces after and before
//      - deal with faces before the base face (all)
//
//  Tags and other inventory assigned here help to expedite and simplify
//  those iterations.
//
void
IrregularPatchBuilder::initializeControlHullInventory() {

    //
    //  Iterate through the corners to identify the vertices, faces and
    //  face-vertices that contribute to the collective control hull --
    //  keeping track of a few situations that cause complications:
    //
    int numVal2IntCorners = 0;
    int numVal3IntAdjTris = 0;
    int numSrcFaceIndices = 0;

    int faceSize = _surface.GetFaceSize();

    _cornerHullInfo.SetSize(faceSize);

    _numControlFaces     = 1;
    _numControlVerts     = faceSize;
    _numControlFaceVerts = faceSize;

    for (int corner = 0; corner < faceSize; ++corner) {
        FaceVertex       const & cTop = _surface.GetCornerTopology(corner);
        FaceVertexSubset const & cSub = _surface.GetCornerSubset(corner);

        //
        //  Inspect faces after the corner face first -- dealing with a few
        //  special cases for interior vertices of low valence -- followed
        //  by those faces before the corner face:
        //
        CornerHull & cHull = _cornerHullInfo[corner];
        cHull.Clear();

        int numCornerFaceVerts = 0;

        if (cSub._numFacesAfter) {
            int nextFace = cTop.GetFaceNext(cTop.GetFace());

            if (cSub.IsBoundary()) {
                //  Boundary -- no special cases:
                for (int i = 1; i < cSub._numFacesAfter; ++i) {
                    nextFace = cTop.GetFaceNext(nextFace);
                    int S = cTop.GetFaceSize(nextFace);

                    cHull.numControlVerts += S - 2;
                    numCornerFaceVerts += S;
                }
                cHull.numControlFaces = cSub._numFacesAfter - 1;
                //  Include unshared vertex of trailing edge
                cHull.numControlVerts ++;
            } else if ((cSub._numFacesTotal == 3) &&
                    (cTop.GetFaceSize(cTop.GetFaceAfter(2)) == 3)) {
                //  Interior, valence-3, adjacent triangle -- special case:
                if (++numVal3IntAdjTris == faceSize) {
                    cHull.singleSharedVert = true;
                    cHull.numControlVerts = 1;
                }
                cHull.numControlFaces = 1;
                numCornerFaceVerts = 3;
            } else if (cSub._numFacesTotal > 2) {
                //  Interior -- general case:
                for (int i = 2; i < cSub._numFacesTotal; ++i) {
                    nextFace = cTop.GetFaceNext(nextFace);
                    int S = cTop.GetFaceSize(nextFace);

                    cHull.numControlVerts += S - 2;
                    numCornerFaceVerts += S;
                }
                cHull.numControlFaces = cSub._numFacesTotal - 2;
                //  Exclude vertex shared with/contributed by next corner
                cHull.numControlVerts --;
            } else {
                //  Interior, valence-2 -- special case:
                if (++numVal2IntCorners == faceSize) {
                    cHull.singleSharedFace = true;
                    cHull.numControlFaces = 1;
                    numCornerFaceVerts = faceSize;
                }
                cHull.isVal2Interior = true;
            }
        }
        if (cSub._numFacesBefore) {
            assert(cSub.IsBoundary());
            int nextFace = cTop.GetFaceFirst(cSub);

            for (int i = 0; i < cSub._numFacesBefore; ++i) {
                int S = cTop.GetFaceSize(nextFace);
                nextFace = cTop.GetFaceNext(nextFace);

                cHull.numControlVerts += S - 2;
                numCornerFaceVerts += S;
            }
            cHull.numControlFaces += cSub._numFacesBefore;
            //  Exclude vertex shared with/contributed by next corner
            cHull.numControlVerts --;
        }

        //  Assign the contributions for this corner:
        cHull.nextControlVert      = _numControlVerts;
        cHull.surfaceIndicesOffset = numSrcFaceIndices;

        _numControlFaces     += cHull.numControlFaces;
        _numControlVerts     += cHull.numControlVerts;
        _numControlFaceVerts += numCornerFaceVerts;

        numSrcFaceIndices += cTop.GetNumFaceVertices();
    }

    //
    //  Use/build a map for the control vertex indices when incident
    //  faces overlap to an extent that makes traversal ill-defined:
    //
    //  Currently a single val-2 interior vertex is handled without
    //  the map in most cases. The presence of more than one val-2
    //  interior vertex leads to convoluted topology traversals that
    //  are avoided by the use of the map.
    //
    _controlFacesOverlap = (numVal2IntCorners > 1);

    if (numVal2IntCorners == 1) {
        //
        //  Identify and inspect the topology at the val-2 corner to
        //  see if it warrants handling with the vertex map. Tag the
        //  neighboring corners accordingly if not:
        //
        for (int corner = 0; corner < faceSize; ++corner) {
            CornerHull & cHull = _cornerHullInfo[corner];

            if (cHull.isVal2Interior) {
                FaceVertex const & cTop = _surface.GetCornerTopology(corner);

                int oppFaceSize = cTop.GetFaceSize(cTop.GetFaceAfter(1));
                if (oppFaceSize == 3) {
                    //  All three corners of the opposite face are corners
                    //  of the base face
                    _controlFacesOverlap = true;
                    break;
                }
                if ((oppFaceSize == 4) && (numVal3IntAdjTris == (faceSize-2))) {
                    //  Possible "pyramid" where a single exterior vertex
                    //  must added as a special case
                    _controlFacesOverlap = true;
                    break;
                }

                //  Tag the preceding corner of the val-2 corner:
                int prevCorner = corner ? (corner - 1) : (faceSize - 1);
                _cornerHullInfo[prevCorner].preVal2Interior = true;
                break;
            }
        }

        //
        //  If no significant overlap is present, adjust the control vertex
        //  counts for those preceding the val-2 corner (decrementing by 1)
        //  and adjust control vertex offsets for all that follow:
        //
        if (!_controlFacesOverlap) {
            _numControlVerts = faceSize;
            for (int corner = 0; corner < faceSize; ++corner) {
                CornerHull & cHull = _cornerHullInfo[corner];

                cHull.nextControlVert  = _numControlVerts;
                cHull.numControlVerts -= cHull.preVal2Interior;

                _numControlVerts += cHull.numControlVerts;
            }
        }
    }

    _useControlVertMap = _controlFacesOverlap;
    if (_useControlVertMap) {
        initializeControlVertexMap();
    }
}

void
IrregularPatchBuilder::addMeshControlVertex(Index meshVertIndex) {

    if (_controlVertMap.find(meshVertIndex) == _controlVertMap.end()) {
        int newLocalVertIndex = (int) _controlVerts.size();
        _controlVertMap[meshVertIndex] = newLocalVertIndex;
        _controlVerts.push_back(meshVertIndex);
    }
}

void
IrregularPatchBuilder::addMeshControlVertices(Index const fVerts[], int fSize) {

    //  Ignore the first index of the face, which corresponds to a corner
    for (int i = 1; i < fSize; ++i) {
        addMeshControlVertex(fVerts[i]);
    }
}

void
IrregularPatchBuilder::initializeControlVertexMap() {

    //
    //  Add CV indices from the base face first -- be careful to ensure
    //  that a vector entry is made for each base face vertex in cases
    //  when repeated indices may occur:
    Index const * baseVerts = getBaseFaceIndices();

    int faceSize = _surface.GetFaceSize();
    for (int i = 0; i < faceSize; ++i) {
        addMeshControlVertex(baseVerts[i]);
        if ((int)_controlVerts.size() == i) {
            _controlVerts.push_back(baseVerts[i]);
        }
    }

    //
    //  For each corner, add face-vertices to the map only for those
    //  incident faces that contribute to the control hull:
    //
    for (int corner = 0; corner < faceSize; ++corner) {
        CornerHull & cHull = _cornerHullInfo[corner];
        if (cHull.numControlFaces == 0) continue;

        FaceVertex       const & cTop = _surface.GetCornerTopology(corner);
        FaceVertexSubset const & cSub = _surface.GetCornerSubset(corner);

        //  Special case of a single shared back-to-back face first:
        if (cHull.singleSharedFace) {
            int nextFace = cTop.GetFaceAfter(1);
            addMeshControlVertices(getCornerFaceIndices(corner, nextFace),
                                   cTop.GetFaceSize(nextFace));
            continue;
        }

        //  Follow the common pattern:  faces after, boundary, faces before
        //  (no need to deal with isolated boundary vertex in this case)
        if (cSub._numFacesAfter > 1) {
            int nextFace = cTop.GetFaceAfter(1);
            for (int j = 1; j < cSub._numFacesAfter; ++j) {
                nextFace = cTop.GetFaceNext(nextFace);

                addMeshControlVertices(getCornerFaceIndices(corner, nextFace),
                                       cTop.GetFaceSize(nextFace));
            }
        }
        if (cSub._numFacesBefore) {
            int nextFace = cTop.GetFaceFirst(cSub);
            for (int i = 0; i < cSub._numFacesBefore; ++i) {
                addMeshControlVertices(getCornerFaceIndices(corner, nextFace),
                                       cTop.GetFaceSize(nextFace));

                nextFace = cTop.GetFaceNext(nextFace);
            }
        }
    }
    _numControlVerts = (int) _controlVerts.size();
}

//
//  Methods for gathering control vertices, faces, sharpness, etc. -- the
//  method to gather control vertex indices is for external use, while the
//  rest are internal:
//
int
IrregularPatchBuilder::GatherControlVertexIndices(Index cvIndices[]) const {

    //
    //  If a map was built, simply copy the associated vector of indices:
    //
    if (_useControlVertMap) {
        std::memcpy(cvIndices, &_controlVerts[0], _numControlVerts*sizeof(int));
        return _numControlVerts;
    }

    //
    //  Assign CV indices from the base face first:
    //
    int faceSize   = _surface.GetFaceSize();
    int numIndices = faceSize;

    std::memcpy(cvIndices, getBaseFaceIndices(), faceSize * sizeof(Index));

    //
    //  Assign vertex indices identified as contributed by each corner:
    //
    for (int corner = 0; corner < faceSize; ++corner) {
        CornerHull const & cHull = _cornerHullInfo[corner];
        if (cHull.numControlVerts == 0) continue;

        FaceVertex       const & cTop   = _surface.GetCornerTopology(corner);
        FaceVertexSubset const & cSub   = _surface.GetCornerSubset(corner);

        //  Special case with all val-3 interior triangles:
        if (cHull.singleSharedVert) {
            assert(!cSub.IsBoundary() && (cSub._numFacesTotal == 3) &&
                   (cTop.GetFaceSize(cTop.GetFaceAfter(2)) == 3));

            cvIndices[numIndices++] =
                    getCornerFaceIndices(corner, cTop.GetFaceAfter(2))[1];
            continue;
        }

        //
        //  Follow the common pattern:  faces after, boundary, faces before
        //
        if (cSub._numFacesAfter > 1) {
            int nextFace = cTop.GetFaceAfter(1);
            int N = cSub._numFacesAfter - 1;
            for (int j = 0; j < N; ++j) {
                nextFace = cTop.GetFaceNext(nextFace);

                Index const * faceVerts = getCornerFaceIndices(corner,nextFace);

                int S = cTop.GetFaceSize(nextFace);
                int L = (j == (N-1)) ? (1 + cHull.preVal2Interior) : 0;
                int M = (S - 2) - (cSub.IsBoundary() ? 0 : L);
                for (int k = 1; k <= M; ++k) {
                    cvIndices[numIndices++] = faceVerts[k];
                }
            }
        }
        if (cSub._numFacesAfter && cSub.IsBoundary()) {
            //  Include trailing edge for boundary before crossing the gap:
            int nextFace = cTop.GetFaceAfter(cSub._numFacesAfter);
            cvIndices[numIndices++] = cTop.GetFaceIndexTrailing(nextFace,
                                                    getCornerIndices(corner));
        }
        if (cSub._numFacesBefore) {
            int nextFace = cTop.GetFaceFirst(cSub);
            int N = cSub._numFacesBefore;
            for (int j = 0; j < N; ++j) {
                Index const * faceVerts = getCornerFaceIndices(corner,nextFace);

                int S = cTop.GetFaceSize(nextFace);
                int L = (j == (N-1)) ? (1 + cHull.preVal2Interior) : 0;
                int M = (S - 2) - L;
                for (int k = 1; k <= M; ++k) {
                    cvIndices[numIndices++] = faceVerts[k];
                }
                nextFace = cTop.GetFaceNext(nextFace);
            }
        }
    }
    assert(numIndices == _numControlVerts);
    return numIndices;
}

int
IrregularPatchBuilder::gatherControlFaces(int faceSizes[],
                                          int faceVertices[]) const {

    //
    //  Assign face-vertices for the first/base face:
    //
    int * faceVerts = faceVertices;

    int faceSize = _surface.GetFaceSize();
    for (int i = 0; i < faceSize; ++i) {
        *faceVerts++ = i;
    }
    *faceSizes++ = faceSize;

    //
    //  Assign face-vertex indices for faces "local to" each corner:
    //
    for (int corner = 0; corner < faceSize; ++corner) {
        CornerHull const & cHull = _cornerHullInfo[corner];
        if (cHull.numControlFaces == 0) continue;

        FaceVertex       const & cTop = _surface.GetCornerTopology(corner);
        FaceVertexSubset const & cSub = _surface.GetCornerSubset(corner);

        //  Special case of a single shared opposing face first:
        if (cHull.singleSharedFace) {
            assert(_useControlVertMap);
            getControlFaceVertices(faceVerts, faceSize, corner,
                        getCornerFaceIndices(corner, cTop.GetFaceAfter(1)));
            *faceSizes++ = faceSize;
            faceVerts   += faceSize;
            continue;
        }

        //
        //  Follow the common pattern:  faces after, boundary, faces before
        //
        int nextVert = cHull.nextControlVert;

        if (cSub._numFacesAfter > 1) {
            int nextFace = cTop.GetFaceAfter(2);
            int N = cSub._numFacesAfter - 1;
            for (int j = 0; j < N; ++j) {
                int S = cTop.GetFaceSize(nextFace);

                if (_useControlVertMap) {
                    getControlFaceVertices(faceVerts, S, corner,
                            getCornerFaceIndices(corner, nextFace));
                } else if (cSub.IsBoundary()) {
                    getControlFaceVertices(faceVerts, S, corner, nextVert);
                } else {
                    getControlFaceVertices(faceVerts, S, corner, nextVert,
                            (j == (N-1)), cHull.preVal2Interior);
                }
                *faceSizes++ = S;
                faceVerts   += S;

                nextVert += S - 2 - (cHull.preVal2Interior && (j == (N-1)));
                nextFace  = cTop.GetFaceNext(nextFace);
            }
        }
        if (cSub._numFacesAfter && cSub.IsBoundary()) {
            nextVert ++;
        }
        if (cSub._numFacesBefore) {
            int nextFace = cTop.GetFaceFirst(cSub);
            int N = cSub._numFacesBefore;
            for (int j = 0; j < N; ++j) {
                int S = cTop.GetFaceSize(nextFace);

                if (_useControlVertMap) {
                    getControlFaceVertices(faceVerts, S, corner,
                            getCornerFaceIndices(corner, nextFace));
                } else {
                    getControlFaceVertices(faceVerts, S, corner, nextVert,
                            (j == (N-1)), cHull.preVal2Interior);
                }
                *faceSizes++ = S;
                faceVerts   += S;

                nextVert += S - 2 - (cHull.preVal2Interior && (j == (N-1)));
                nextFace  = cTop.GetFaceNext(nextFace);
            }
        }
    }
    assert((faceVerts - faceVertices) == _numControlFaceVerts);
    return _numControlFaceVerts;
}

int
IrregularPatchBuilder::gatherControlVertexSharpness(
        int vertIndices[], float vertSharpness[]) const {

    int nSharpVerts = 0;
    for (int i = 0; i < _surface.GetFaceSize(); ++i) {
        FaceVertexSubset const & cSub = _surface.GetCornerSubset(i);

        if (cSub._tag.IsInfSharp()) {
            vertSharpness[nSharpVerts] = Sdc::Crease::SHARPNESS_INFINITE;
            vertIndices[nSharpVerts++] = i;
        } else if (cSub._tag.IsSemiSharp()) {
            vertSharpness[nSharpVerts] = (cSub._localSharpness > 0.0f) ? 
                        cSub._localSharpness :
                        _surface.GetCornerTopology(i).GetVertexSharpness();
            vertIndices[nSharpVerts++] = i;
        }
    }
    return nSharpVerts;
}

int
IrregularPatchBuilder::gatherControlEdgeSharpness(
        int edgeVertPairs[], float edgeSharpness[]) const {

    //
    //  First test the forward edge of each corner of the face (avoid
    //  including redundant inf-sharp boundary edges):
    //
    int nSharpEdges = 0;

    int faceSize = _surface.GetFaceSize();

    for (int corner = 0; corner < faceSize; ++corner) {
        FaceVertexSubset const & cSub = _surface.GetCornerSubset(corner);
        if (!cSub._tag.HasSharpEdges()) continue;

        if (!cSub.IsBoundary() || cSub._numFacesBefore) {
            FaceVertex const & cTop = _surface.GetCornerTopology(corner);

            int   cornerFace = cTop.GetFace();
            float sharpness  = cTop.GetFaceEdgeSharpness(cornerFace, 0);
            if (Sdc::Crease::IsSharp(sharpness)) {
                *edgeSharpness++ = sharpness;
                *edgeVertPairs++ =  corner;
                *edgeVertPairs++ = (corner + 1) % faceSize;
                nSharpEdges++;
            }
        }
    }

    //
    //  For each corner, test any interior edges connected to vertices
    //  on the perimeter:
    //
    for (int corner = 0; corner < faceSize; ++corner) {
        FaceVertexSubset const & cSub = _surface.GetCornerSubset(corner);
        if (!cSub._tag.HasSharpEdges()) continue;

        CornerHull const & cHull = _cornerHullInfo[corner];
        if (cHull.numControlFaces == 0) continue;

        FaceVertex const & cTop = _surface.GetCornerTopology(corner);

        //
        //  Inspect interior edges around the subset -- testing sharpness
        //  of the trailing edge of the faces after/before the corner face.
        //
        //  Follow the common pattern:  faces after, boundary, faces before
        //
        //  Track perimeter index to identify verts at end of sharp edges:
        int maxVert  = _numControlVerts;
        int nextVert = cHull.nextControlVert;

        Index const * cVerts = getCornerIndices(corner);

        if (cSub._numFacesAfter > 1) {
            int nextFace = cTop.GetFaceAfter(1);
            for (int i = 1; i < cSub._numFacesAfter; ++i) {
                float sharpness = cTop.GetFaceEdgeSharpness(nextFace, 1);
                if (Sdc::Crease::IsSharp(sharpness)) {
                    int edgeVert = (nextVert < maxVert) ? nextVert : faceSize;
                    if (_useControlVertMap) {
                        edgeVert = getLocalControlVertex(
                            cTop.GetFaceIndexTrailing(nextFace, cVerts));
                    }

                    *edgeSharpness++ = sharpness;
                    *edgeVertPairs++ = corner;
                    *edgeVertPairs++ = edgeVert;
                    nSharpEdges++;
                }
                nextFace  = cTop.GetFaceNext(nextFace);
                nextVert += cTop.GetFaceSize(nextFace) - 2;
            }
        }
        if (cSub._numFacesAfter && cSub.IsBoundary()) {
            nextVert += cSub.IsBoundary();
        }
        if (cSub._numFacesBefore) {
            int nextFace = cTop.GetFaceFirst(cSub);
            for (int i = 1; i < cSub._numFacesBefore; ++i) {
                nextVert += cTop.GetFaceSize(nextFace) - 2;
                float sharpness = cTop.GetFaceEdgeSharpness(nextFace, 1);
                if (Sdc::Crease::IsSharp(sharpness)) {
                    int edgeVert = (nextVert < maxVert) ? nextVert : faceSize;
                    if (_useControlVertMap) {
                        edgeVert = getLocalControlVertex(
                            cTop.GetFaceIndexTrailing(nextFace, cVerts));
                    }

                    *edgeSharpness++ = sharpness;
                    *edgeVertPairs++ = corner;
                    *edgeVertPairs++ = edgeVert;
                    nSharpEdges++;
                }
                nextFace  = cTop.GetFaceNext(nextFace);
            }
        }
    }
    return nSharpEdges;
}


//
//  Methods to gather the local face-vertices for a particular incident
//  face of a corner. The first is a special case that uses the vertex
//  map to identify local control vertices from their indices. The second
//  is the trivial case -- where all vertices other than the initial
//  corner vertex lie on the perimeter and do not wrap around. The third
//  handles more of the complications and is used in all cases where the
//  trivial method cannot be applied.
//
//
void
IrregularPatchBuilder::getControlFaceVertices(int fVerts[], int numFVerts,
        int corner, Index const srcVerts[]) const {
    assert(_useControlVertMap);

    *fVerts++ = corner;
    for (int i = 1; i < numFVerts; ++i) {
        *fVerts++ = getLocalControlVertex(srcVerts[i]);
    }
}

void
IrregularPatchBuilder::getControlFaceVertices(int fVerts[], int numFVerts,
        int corner, int nextPerimeterVert) const {

    *fVerts++ = corner;
    for (int i = 1; i < numFVerts; ++i) {
        *fVerts++ = nextPerimeterVert + i - 1;
    }
}

void
IrregularPatchBuilder::getControlFaceVertices(int fVerts[], int numFVerts,
    int corner, int nextPerimeterVert, bool lastFace, int numVal2InLast) const {

    //
    //  When identifying local face-vertices for a control face adjacent
    //  to a corner, care is required to deal with the possibility of
    //  exterior vertices being shared with control faces associated with
    //  the next corner vertex -- which becomes an issue when sharing
    //  between the last and first corner.
    //
    //  After the initial corner face-vertex, the remaining face-vertices
    //  are identified in three groups:
    //    - a simple sequence of exterior vertices following the corner,
    //      which excludes the last two face-vertices where any sharing
    //      may occur
    //    - the next-to-last face-vertex -- an exterior vertex which may
    //      be shared and wrap around the base face
    //    - the last face vertex -- an exterior vertex which may also be
    //      shared/wrapped, or an adj corner vertex for the last face
    //
    int S = numFVerts;
    int N = _surface.GetFaceSize();

    //  Start with the corner vertex:
    *fVerts++ = corner;

    //  The simple sequence of exterior vertices follows the corner:
    int numSequentialVerts = S - 2 - 1 - (lastFace ? numVal2InLast : 0);
    for (int i = 0; i < numSequentialVerts; ++i) {
        *fVerts++ = nextPerimeterVert++;
    }

    //  The next-to-last face-vertex may be shared with the next face
    //  and so may wrap around the entire face:
    int nextToLastPerimOfFace = nextPerimeterVert++;
    if (nextToLastPerimOfFace == _numControlVerts) {
        nextToLastPerimOfFace = N;
    }
    *fVerts++ = nextToLastPerimOfFace;

    //  The last face-vertex may be exterior or the adjacent corner:
    if (!lastFace) {
        //  If exterior, it may also be shared with the next face and
        //  wrap around the entire face:
        int lastPerimOfFace = nextPerimeterVert++;
        if (lastPerimOfFace == _numControlVerts) {
            lastPerimOfFace = N;
        }
        *fVerts = lastPerimOfFace;
    } else {
        //  If the next/adjacent corner, successive corner vertices may
        //  need to be added if that corner is a val-2 interior vertex:
        for (int i = numVal2InLast; i > 0; --i) {
            *fVerts++ = (corner + 1 + i) % N;
        }
        *fVerts = (corner + 1) % N;
    }
}

//
//  Detection and removal of duplicate control faces -- which can only
//  occur when incident faces overlap with the base face:
//
namespace {
    //
    //  Internal helper functions to detect duplicate faces:
    //
    bool
    doFacesMatch(int size, int const a[], int const b[], int bStart) {
        for (int i = 0, j = bStart; i < size; ++i, ++j) {
            j = (j == size) ? 0 : j;
            if (a[i] != b[j]) return false;
        }
        return true;
    }

    bool
    doFacesMatch(int size, int const a[], int const b[]) {
        //  Find a matching vertex to correlate possible rotation:
        for (int i = 0; i < size; ++i) {
            if (b[i] == a[0]) {
                return doFacesMatch(size, a, b, i);
            }
        }
        return false;
    }
}

void
IrregularPatchBuilder::removeDuplicateControlFaces(
        int   faceSizes[], int   faceVerts[],
        int * numFaces,    int * numFaceVerts) const {

    //
    //  Work backwards from the last face -- detecting if it matches a
    //  face earlier in the arrays and removing it if so:
    //
    int numSizesAfter = 0;
    int numVertsAfter = 0;

    int * sizesAfter = faceSizes + *numFaces;
    int * vertsAfter = faceVerts + *numFaceVerts;

    for (int i = *numFaces - 1; i > 1; --i) {
        int   iSize  = faceSizes[i];
        int * iVerts = vertsAfter - iSize;

        //  Inspect the faces preceding this face for a duplicate:
        bool isDuplicate = false;

        int * jVerts = iVerts;
        for (int j = i - 1; !isDuplicate && (j > 0); --j) {
            jVerts = jVerts - faceSizes[j];
            if (iSize == faceSizes[j]) {
                isDuplicate = doFacesMatch(iSize, iVerts, jVerts);
            }
        }

        //  If this face was duplicated by one preceding it, remove it:
        if (isDuplicate) {
            if (numSizesAfter) {
                std::memmove(sizesAfter - 1, sizesAfter,
                             numSizesAfter * sizeof(int));
                std::memmove(vertsAfter - iSize, vertsAfter,
                             numVertsAfter * sizeof(int));
            }
            (*numFaces) --;
            (*numFaceVerts) -= iSize;
        } else {
            numSizesAfter ++;
            numVertsAfter += iSize;
        }
        sizesAfter --;
        vertsAfter -= iSize;
    }
}

void
IrregularPatchBuilder::sharpenBoundaryControlEdges(
        int edgeVertPairs[], float edgeSharpness[], int * numSharpEdges) const {

    //
    //  When extracting a manifold subset from a greater non-manifold
    //  region, the boundary edges for the subset sometimes occur on
    //  non-manifold edges.  When extracting the subset topology in
    //  cases where faces overlap, those boundary edges can sometimes
    //  be misinterpreted as manifold interior edges -- as the extra
    //  faces connected to them that made the edge non-manifold are
    //  not included in the subset.
    //
    //  So boundary edges are sharpened here -- which generally has no
    //  effect (as boundary edges are implicitly sharpened) but ensures
    //  that if the edge is misinterpreted as interior, it will remain
    //  sharp. And only boundary edges of the base face are sharpened
    //  here -- it is not necessary to deal with others.
    //
    //  Append boundary edge sharpness to existing sharp edges:
    edgeVertPairs += *numSharpEdges * 2;
    edgeSharpness += *numSharpEdges;

    int faceSize = _surface.GetFaceSize();

    for (int corner = 0; corner < faceSize; ++corner) {
        FaceVertexSubset const & cSub = _surface.GetCornerSubset(corner);

        if (cSub.IsBoundary() && (cSub._numFacesBefore == 0)) {
            *edgeSharpness++ = Sdc::Crease::SHARPNESS_INFINITE;
            *edgeVertPairs++ =  corner;
            *edgeVertPairs++ = (corner + 1) % faceSize;
            (*numSharpEdges) ++;
        }
    }
}

//
//  The main build/assembly method to create a PatchTree:
//
//  Note that the IrregularPatchBuilder was conceived to potentially build
//  different representations of irregular patches (all sharing a virtual
//  interface to hide that from it clients).  It was here that topology
//  would be inspected and builders for the different representations would
//  be dispatched.
//
//  At this point, the PatchTree is used for all topological cases, so
//  those future intentions are not reflected here.
//
internal::IrregularPatchSharedPtr
IrregularPatchBuilder::Build() {

    //
    //  Build a PatchTree -- the sole representation used for all irregular
    //  patch topologies.
    //
    //  Given the PatchTree's origin in Far, it's builder class requires a
    //  Far::TopologyRefiner.  Now that PatchTreeBuilder is part of Bfr, it
    //  could be adapted to accept the Bfr::FaceSurface directly and deal
    //  with any any intermediate TopologyRefiner internally.
    //
    //  For now, the quickest way of constructing a TopologyRefiner is via
    //  a Far::TopologyDescriptor -- which simply refers to pre-allocated
    //  topology arrays. Those arrays will be allocated on the stack here
    //  to accommodate typical cases not involving excessively high valence.
    //
    //  Use of TopologyDescriptor could be eliminated by defining a factory
    //  to create a TopologyRefiner directly from the FaceSurface, but the
    //  benefits relative to the cost of creating the PatchTree are not
    //  significant. The fact that the current assembly requires removing
    //  duplicate faces in some cases further complicates that process.
    //
    int numVerts     = _numControlVerts;
    int numFaces     = _numControlFaces;
    int numFaceVerts = _numControlFaceVerts;
    int numCorners   = _surface.GetFaceSize();
    int numCreases   = _numControlVerts;

    //  Allocate and partition stack buffers for the topology arrays:
    int numInts   = numFaces + numFaceVerts + numCorners + numCreases*2;
    int numFloats = numCorners + numCreases;

    Vtr::internal::StackBuffer<int, 256,true> intBuffer(numInts);
    Vtr::internal::StackBuffer<float,64,true> floatBuffer(numFloats);

    int * faceSizes     = intBuffer;
    int * faceVerts     = faceSizes     + numFaces;
    int * cornerIndices = faceVerts     + numFaceVerts;
    int * creaseIndices = cornerIndices + numCorners;

    float * cornerWeights = floatBuffer;
    float * creaseWeights = cornerWeights + numCorners;

    //  Gather face topology (sizes and vertices) and explicit sharpness:
    gatherControlFaces(faceSizes, faceVerts);

    numCorners = _surface.GetTag().HasSharpVertices() ?
                 gatherControlVertexSharpness(cornerIndices, cornerWeights) : 0;

    numCreases = _surface.GetTag().HasSharpEdges() ?
                 gatherControlEdgeSharpness(creaseIndices, creaseWeights) : 0;

    //  Make some adjustments when control faces may overlap:
    if (controlFacesMayOverlap()) {
        if (numFaces > 2) {
            removeDuplicateControlFaces(faceSizes, faceVerts,
                                        &numFaces, &numFaceVerts);
        }
        if (_surface.GetTag().HasBoundaryVertices()) {
            sharpenBoundaryControlEdges(creaseIndices, creaseWeights,
                                        &numCreases);
        }
    }

    //  Declare a TopologyDescriptor to reference the data gathered above:
    Far::TopologyDescriptor topDescriptor;

    topDescriptor.numVertices = numVerts;
    topDescriptor.numFaces    = numFaces;

    topDescriptor.numVertsPerFace    = faceSizes;
    topDescriptor.vertIndicesPerFace = faceVerts;

    if (numCorners) {
        topDescriptor.numCorners          = numCorners;
        topDescriptor.cornerVertexIndices = cornerIndices;
        topDescriptor.cornerWeights       = cornerWeights;
    }

    if (numCreases) {
        topDescriptor.numCreases             = numCreases;
        topDescriptor.creaseVertexIndexPairs = creaseIndices;
        topDescriptor.creaseWeights          = creaseWeights;
    }

    //  Construct a TopologyRefiner in order to create a PatchTree:
    typedef Far::TopologyDescriptor Descriptor;
    typedef Far::TopologyRefinerFactory<Descriptor> RefinerFactory;

    RefinerFactory::Options refinerOptions;
    refinerOptions.schemeType    = _surface.GetSdcScheme();
    refinerOptions.schemeOptions = _surface.GetSdcOptionsInEffect();
    // WIP - enable for debugging
    //refinerOptions.validateFullTopology = true;

    Far::TopologyRefiner * refiner =
            RefinerFactory::Create(topDescriptor, refinerOptions);

    //  Create the PatchTree from the TopologyRefiner:
    PatchTreeBuilder::Options patchTreeOptions;
    patchTreeOptions.includeInteriorPatches = false;
    patchTreeOptions.maxPatchDepthSharp  = (unsigned char)_options.sharpLevel;
    patchTreeOptions.maxPatchDepthSmooth = (unsigned char)_options.smoothLevel;
    patchTreeOptions.useDoublePrecision  = _options.doublePrecision;

    PatchTreeBuilder patchTreeBuilder(*refiner, patchTreeOptions);

    PatchTree const * patchTree = patchTreeBuilder.Build();

    assert(patchTree->GetNumControlPoints() == _numControlVerts);

    delete refiner;
    return internal::IrregularPatchSharedPtr(patchTree);
}

} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
