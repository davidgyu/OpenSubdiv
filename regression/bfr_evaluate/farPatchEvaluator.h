//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "./types.h"

#include <opensubdiv/far/topologyRefiner.h>
#include <opensubdiv/far/patchTable.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/ptexIndices.h>

#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/tessellation.h>

using namespace OpenSubdiv;
using namespace OpenSubdiv::OPENSUBDIV_VERSION;


//
//  FarPatchEvaluator bundles the Far::PatchTable and its ecosystem of
//  related class to provide an evaluation interface targeted towards
//  evaluation of the base faces of a mesh.
//
template <typename REAL>
class FarPatchEvaluator {
public:
    typedef Far::PatchTableFactory       PatchFactory;
    typedef Bfr::SurfaceFactory::Options BfrSurfaceOptions;

    typedef std::vector<REAL>            TessCoordVector;

    typedef Vec3<REAL>                   Vec3Real;
    typedef std::vector<Vec3Real>        Vec3RealVector;

public:
    FarPatchEvaluator(Far::TopologyRefiner const & baseMesh,
                      Vec3RealVector       const & basePos,
                      Vec3RealVector       const & baseUVs,
                      BfrSurfaceOptions    const & bfrSurfaceOptions);
    ~FarPatchEvaluator();

public:
    bool FaceHasLimit(Far::Index baseFace) const;

    void Evaluate(Far::Index              baseface,
                  TessCoordVector const & tessCoords,
                  EvalResults<REAL>     & results) const;

private:
    Far::TopologyRefiner const & _baseMesh;
    Vec3RealVector       const & _baseMeshPos;
    Vec3RealVector       const & _baseMeshUVs;

    Far::PatchTable  * _patchTable;
    Far::PatchMap    * _patchMap;
    Far::PtexIndices * _patchFaces;

    Vec3RealVector     _patchPos;
    Vec3RealVector     _patchUVs;

    int  _regFaceSize;
};
