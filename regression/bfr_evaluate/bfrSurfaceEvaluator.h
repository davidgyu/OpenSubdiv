//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "./types.h"

#include <opensubdiv/far/topologyRefiner.h>

#include <opensubdiv/bfr/refinerSurfaceFactory.h>
#include <opensubdiv/bfr/tessellation.h>


using namespace OpenSubdiv;
using namespace OpenSubdiv::OPENSUBDIV_VERSION;



template <typename REAL>
class BfrSurfaceEvaluator {
public:
    typedef Bfr::Surface<REAL>           SurfaceType;

    typedef Bfr::RefinerSurfaceFactory<> SurfaceFactory;
    typedef Bfr::SurfaceFactory::Options FactoryOptions;
    typedef Bfr::SurfaceFactory::Index   IndexType;

    typedef std::vector<REAL>            TessCoordVector;

    typedef std::vector< Vec3<REAL> > Vec3Vector;

public:
    BfrSurfaceEvaluator(Far::TopologyRefiner const & baseMesh,
                        Vec3Vector           const & basePos,
                        Vec3Vector           const & baseUVs,
                        FactoryOptions       const & factoryOptions);
    ~BfrSurfaceEvaluator() { }

public:
    bool FaceHasLimit(IndexType baseFace) const;

    void Evaluate(IndexType               baseface,
                  TessCoordVector const & tessCoords,
                  EvalResults<REAL>     & results) const;

private:
    void evaluateDirectly(SurfaceType     const & posSurface,
                          SurfaceType     const & uvSurface,
                          TessCoordVector const & tessCoords,
                          EvalResults<REAL>     & results) const;

    void evaluateByStencils(SurfaceType     const & posSurface,
                            SurfaceType     const & uvSurface,
                            TessCoordVector const & tessCoords,
                            EvalResults<REAL>     & results) const;

private:
    Far::TopologyRefiner const & _baseMesh;
    Vec3Vector           const & _baseMeshPos;
    Vec3Vector           const & _baseMeshUVs;

    SurfaceFactory _factory;
};
