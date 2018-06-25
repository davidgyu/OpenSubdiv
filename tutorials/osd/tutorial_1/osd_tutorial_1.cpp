//
//   Copyright 2016 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//


//------------------------------------------------------------------------------
// Tutorial description:
//
// This tutorial demonstrates the use of the Osd Evaluator API to compute
// limit surface geometry on a tessellated surface.
//
// Prints the resulting tessellated mesh to standard output.
//
// usage:
//  --adaptive         : use feature adaptive refinement.
//  --uniform          : use uniform refinement.
//  --normals          : compute adaptive normals.
//  --fvar             : compute adaptive fvar values.
//  --maxlevel <int>   : topological refinement level.
//  --tesslevel <int>  : geometric tessellation level.
//  --shape <filepath> : read .obj file.
//

// 2016/01/28 dyu example of cpu adaptive tessellation
// 2016/03/24 dyu added evaluation of face-varying values
// 2016/11/30 dyu added stencil refinement of face-varying values

#include "../../../regression/common/far_utils.h"
#include "../../../regression/common/shape_utils.h"

#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/patchMap.h>
#include <opensubdiv/far/patchTable.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/stencilTableFactory.h>
#include <opensubdiv/far/topologyRefinerFactory.h>
#include <opensubdiv/osd/cpuEvaluator.h>
#include <opensubdiv/osd/cpuPatchTable.h>
#include <opensubdiv/osd/cpuVertexBuffer.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace OpenSubdiv;

static std::string modelName;
static std::vector<float> basePoints;
static std::vector<float> baseUVs;

//------------------------------------------------------------------------------
// Default cube geometry from catmark_cube.h

static float g_verts[8*3] = {-0.5f, -0.5f,  0.5f,
                              0.5f, -0.5f,  0.5f,
                             -0.5f,  0.5f,  0.5f,
                              0.5f,  0.5f,  0.5f,
                             -0.5f,  0.5f, -0.5f,
                              0.5f,  0.5f, -0.5f,
                             -0.5f, -0.5f, -0.5f,
                              0.5f, -0.5f, -0.5f };

static int g_nverts = 8,
           g_nfaces = 6;

static int g_vertsperface[6] = { 4, 4, 4, 4, 4, 4 };

static int g_vertIndices[6*4] = { 0, 1, 3, 2,
                                  2, 3, 5, 4,
                                  4, 5, 7, 6,
                                  6, 7, 1, 0,
                                  1, 7, 5, 3,
                                  6, 0, 2, 4 };

// 'face-varying' primitive variable data & topology for UVs
static float g_uvs[14*2] = { 0.375, 0.00,
                             0.625, 0.00,
                             0.375, 0.25,
                             0.625, 0.25,
                             0.375, 0.50,
                             0.625, 0.50,
                             0.375, 0.75,
                             0.625, 0.75,
                             0.375, 1.00,
                             0.625, 1.00,
                             0.875, 0.00,
                             0.875, 0.25,
                             0.125, 0.00,
                             0.125, 0.25 };

static int g_nuvs = 14;

static int g_uvIndices[6*4] = {  0,  1,  3,  2,
                                 2,  3,  5,  4,
                                 4,  5,  7,  6,
                                 6,  7,  9,  8,
                                 1, 10, 11,  3,
                                12,  0,  2, 13  };

//------------------------------------------------------------------------------
// Create topology refiner for default cube mesh

static Far::TopologyRefiner *
createDefaultTopologyRefiner() {

    typedef Far::TopologyDescriptor Descriptor;

    Descriptor desc;
    desc.numVertices = g_nverts;
    desc.numFaces = g_nfaces;
    desc.numVertsPerFace = g_vertsperface;
    desc.vertIndicesPerFace = g_vertIndices;

    // Create a face-varying channel descriptor
    const int numFVarChannels = 1;
    const int uvChannel = 0;
    Descriptor::FVarChannel channels[numFVarChannels];
    channels[uvChannel].numValues = g_nuvs;
    channels[uvChannel].valueIndices = g_uvIndices;

    // Add the channel topology to the main descriptor
    desc.numFVarChannels = numFVarChannels;
    desc.fvarChannels = channels;

    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_ONLY);

    // Instantiate a FarTopologyRefiner from the descriptor
    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Descriptor>::Create(desc,
            Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    modelName = "defaultCube";
    basePoints.assign(&g_verts[0], &g_verts[3*g_nverts]);
    baseUVs.assign(&g_uvs[0], &g_uvs[2*g_nuvs]);

    return refiner;
}

//------------------------------------------------------------------------------
// Create topology refiner for geometry read from obj file

static Far::TopologyRefiner *
createTopologyRefiner(std::string const & filepath) {

    std::ifstream ifs(filepath);
    if (! ifs) {
        return NULL;
    }

    std::stringstream ss;
    ss << ifs.rdbuf();
    ifs.close();

    std::string shapeString = ss.str();
    bool isLeftHanded = false;
    Shape const * shape =
        Shape::parseObj(shapeString.c_str(), kCatmark, isLeftHanded);

    Sdc::SchemeType sdctype = GetSdcType(*shape);
    Sdc::Options sdcoptions = GetSdcOptions(*shape);

    // Instantiate a FarTopologyRefiner from the Shape descriptor
    Far::TopologyRefiner * refiner =
        Far::TopologyRefinerFactory<Shape>::Create(*shape,
            Far::TopologyRefinerFactory<Shape>::Options(sdctype, sdcoptions));

    modelName = filepath;
    basePoints.assign(shape->verts.begin(), shape->verts.end());
    baseUVs.assign(shape->uvs.begin(), shape->uvs.end());

    return refiner;
}

//------------------------------------------------------------------------------
// Simple Tessellator

class Tessellator {
public:
    Tessellator(Far::TopologyRefiner const * refiner,
                Far::PatchTable const * patchTable,
                int tesslevel);

    //
    // Locations of vertices to evaluate on the tessellated surface.
    //

    int GetNumPatchCoords() const {
        return (int)_patchCoords.size();
    }
    Osd::PatchCoord const * GetPatchCoords() const {
        return &_patchCoords[0];
    }

    //
    // Topology of the tessellated surface.
    //

    int GetNumFaceVertexIndices() const {
        return (int)_faceIndices.size();
    }
    int const * GetFaceVertexIndices() const {
        return &_faceIndices[0];
    }

    int GetNumFaceVertexCounts() const {
        return (int)_faceCounts.size();
    }
    int const * GetFaceVertexCounts() const {
        return &_faceCounts[0];
    }

private:
    struct PatchCoordKey {
        PatchCoordKey(Osd::PatchCoord const & coord)
            : arrayIndex(coord.handle.arrayIndex)
            , patchIndex(coord.handle.patchIndex)
            , vertIndex(coord.handle.vertIndex)
            , s(coord.s), t(coord.t) { }
        int arrayIndex, patchIndex, vertIndex;
        float s, t;
    };
    struct PatchCoordKeyCompare {
        bool operator()(PatchCoordKey const & lhs,
                        PatchCoordKey const & rhs) const {

            return (lhs.arrayIndex < rhs.arrayIndex ||
                       (lhs.arrayIndex == rhs.arrayIndex &&
                    (lhs.patchIndex < rhs.patchIndex ||
                        (lhs.patchIndex == rhs.patchIndex &&
                     (lhs.vertIndex < rhs.vertIndex ||
                         (lhs.vertIndex == rhs.vertIndex &&
                      (lhs.s < rhs.s ||
                          (lhs.s == rhs.s && (lhs.t < rhs.t)))))))));
        }
    };
    typedef std::map<PatchCoordKey, int, PatchCoordKeyCompare> PatchCoordMap;

    struct TessBuilder {
        Far::PatchMap const * patchMap;
        PatchCoordMap patchCoordMap;
    };

    void _Tessellate(Far::PatchTable const * patchTable, int tesslevel);

    void _AddPatch(TessBuilder * tessBuilder,
                   Far::PatchParam const & patchParam, int tesslevel);

    void _AddPatchGrid(TessBuilder * tessBuilder,
                       Far::PatchParam const & patchParam, int gridSize);

    void _AddPatchSingle(TessBuilder * tessBuilder,
                         Far::PatchParam const & patchParam);

    int _GetPatchCoordIndex(TessBuilder * tessBuilder,
                            Far::PatchParam const & patchParam,
                            float u, float v);

    Osd::PatchCoord _GetPatchCoord(Far::PatchMap const * patchMap,
                                   Far::PatchParam const & patchParam,
                                   float u, float v);

    void _AddTriangle(int v0, int v1, int v2);

private:
    std::vector<Osd::PatchCoord> _patchCoords;
    std::vector<int> _faceIndices;
    std::vector<int> _faceCounts;
};

Tessellator::Tessellator(
        Far::TopologyRefiner const * /*refiner*/,
        Far::PatchTable const * patchTable,
        int tesslevel) {

    // XXX refiner is unused, but could help eliminated duplicated points.

    if (patchTable->IsFeatureAdaptive()) {
        _Tessellate(patchTable, tesslevel);
    }
}

void
Tessellator::_Tessellate(Far::PatchTable const * patchTable,
                         int tesslevel) {

    TessBuilder tessBuilder;
    tessBuilder.patchMap = new Far::PatchMap(*patchTable);

    // Tessellate each patch.
    for (int array=0; array<patchTable->GetNumPatchArrays(); ++array) {
        for (int patch=0; patch<patchTable->GetNumPatches(array); ++patch) {
            _AddPatch(&tessBuilder,
                      patchTable->GetPatchParam(array, patch), tesslevel);
        }
    }

    delete tessBuilder.patchMap;
}

void
Tessellator::_AddPatch(TessBuilder * tessBuilder,
                       Far::PatchParam const & patchParam,
                       int tesslevel) {

    // Determine the grid size for each patch based on its level of refinement.
    // Constrain grid sizes to power's of two to prevent cracks.
    int patchRefinementLevel = patchParam.GetDepth();
    int gridSize = int((1 << tesslevel) / pow(2, patchRefinementLevel-1));

    if (gridSize > 1) {
        _AddPatchGrid(tessBuilder, patchParam, gridSize);
    } else {
        _AddPatchSingle(tessBuilder, patchParam);
    }
}

void
Tessellator::_AddPatchGrid(TessBuilder * tessBuilder,
                           Far::PatchParam const & patchParam,
                           int gridSize) {

    // The grid of points needed for gridSize x gridSize quads.
    int pointGridSize = gridSize + 1;
    std::vector<int> p(pointGridSize*pointGridSize,0);

    for (int i=0; i<=gridSize; ++i) {
        for (int j=0; j<=gridSize; ++j) {
            int pointIndex = i * pointGridSize + j;
            float u = float(j)/gridSize;
            float v = float(i)/gridSize;
            p[pointIndex] = _GetPatchCoordIndex(tessBuilder, patchParam, u, v);
        }
    }

    // Emit two triangles for each quad covering the grid of points.
    for (int i=0; i<gridSize; ++i) {
        for (int j=0; j<gridSize; ++j) {
            int pointIndex = i * pointGridSize + j;
            int p0 = p[pointIndex];
            int p1 = p[pointIndex+1];
            int p2 = p[pointIndex+pointGridSize+1];
            int p3 = p[pointIndex+pointGridSize];

            // Flip quad triangulation to produce a symmetrical tessellation.
            if ((i < gridSize/2) ^ (j < gridSize/2)) {
                _AddTriangle(p0, p1, p2);
                _AddTriangle(p2, p3, p0);
            } else {
                _AddTriangle(p0, p1, p3);
                _AddTriangle(p2, p3, p1);
            }
        }
    }
}

void
Tessellator::_AddPatchSingle(TessBuilder * tessBuilder,
                             Far::PatchParam const & patchParam) {

    // The four corners of the patch
    int p[4] = { _GetPatchCoordIndex(tessBuilder, patchParam, 0.0f, 0.0f),
                 _GetPatchCoordIndex(tessBuilder, patchParam, 1.0f, 0.0f),
                 _GetPatchCoordIndex(tessBuilder, patchParam, 1.0f, 1.0f),
                 _GetPatchCoordIndex(tessBuilder, patchParam, 0.0f, 1.0f) };

    int transitionMask = patchParam.GetTransition();

    // When there are no transition edges, just emit two triangles.
    if (transitionMask == 0) {
        _AddTriangle(p[0], p[1], p[2]);
        _AddTriangle(p[2], p[3], p[0]);
        return;
    }

    // Handle transition cases by emitting 5 to 8 triangles. Evaluating
    // the center of the patch and the midPoint of each transition edge.
    int center = _GetPatchCoordIndex(tessBuilder, patchParam, 0.5f, 0.5f);
    if (transitionMask & 1) {
        int midPoint = _GetPatchCoordIndex(tessBuilder, patchParam, 0.5f, 0.0f);
        _AddTriangle(p[0], midPoint, center);
        _AddTriangle(midPoint, p[1], center);
    } else {
        _AddTriangle(p[0], p[1], center);
    }
    if (transitionMask & 2) {
        int midPoint = _GetPatchCoordIndex(tessBuilder, patchParam, 1.0f, 0.5f);
        _AddTriangle(p[1], midPoint, center);
        _AddTriangle(midPoint, p[2], center);
    } else {
        _AddTriangle(p[1], p[2], center);
    }
    if (transitionMask & 4) {
        int midPoint = _GetPatchCoordIndex(tessBuilder, patchParam, 0.5f, 1.0f);
        _AddTriangle(p[2], midPoint, center);
        _AddTriangle(midPoint, p[3], center);
    } else {
        _AddTriangle(p[2], p[3], center);
    }
    if (transitionMask & 8) {
        int midPoint = _GetPatchCoordIndex(tessBuilder, patchParam, 0.0f, 0.5f);
        _AddTriangle(p[3], midPoint, center);
        _AddTriangle(midPoint, p[0], center);
    } else {
        _AddTriangle(p[3], p[0], center);
    }
}

int
Tessellator::_GetPatchCoordIndex(TessBuilder * tessBuilder,
                                 Far::PatchParam const & patchParam,
                                 float u, float v) {

    Osd::PatchCoord patchCoord =
        _GetPatchCoord(tessBuilder->patchMap, patchParam, u, v);

    // Use patchCoordMap to prevent evaluation of duplicate points within
    // each coarse face. We could make use of adjacency information from the
    // TopologyRefiner to avoid duplicate points between coarse faces.
    PatchCoordKey key(patchCoord);
    PatchCoordMap::const_iterator it = tessBuilder->patchCoordMap.find(key);
    if (it != tessBuilder->patchCoordMap.end()) {
        return it->second;
    }

    int patchCoordIndex = (int)_patchCoords.size();
    _patchCoords.push_back(patchCoord);

    tessBuilder->patchCoordMap[key] = patchCoordIndex;
    return patchCoordIndex;
}

Osd::PatchCoord
Tessellator::_GetPatchCoord(Far::PatchMap const * patchMap,
                            Far::PatchParam const & patchParam,
                            float u, float v) {

    // Use patchMap to find the PatchHandle for the sample position and
    // return the corresponding PatchCoord.
    patchParam.Unnormalize(u, v);
    Far::PatchTable::PatchHandle const *
        handle = patchMap->FindPatch(patchParam.GetFaceId(), u, v);
    return Osd::PatchCoord(*handle, u, v);
}

void
Tessellator::_AddTriangle(int v0, int v1, int v2) {

    // Add a triangle to the tessellated topology.
    _faceIndices.push_back(v0);
    _faceIndices.push_back(v1);
    _faceIndices.push_back(v2);
    _faceCounts.push_back(3);
}

static void
computeNormal(float *n, const float *du, const float *dv) {

    n[0] = du[1]*dv[2]-du[2]*dv[1];
    n[1] = du[2]*dv[0]-du[0]*dv[2];
    n[2] = du[0]*dv[1]-du[1]*dv[0];

    float rn = 1.0f/sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    n[0] *= rn;
    n[1] *= rn;
    n[2] *= rn;
}

//------------------------------------------------------------------------------
int main(int argc, char **argv) {

    bool refineAdaptive = true;
    bool refineFVar = false;
    bool computeNormals = false;
    int maxlevel = 2;
    int tesslevel = 1;
    std::string filepath;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--adaptive")) {
            refineAdaptive = true;
        } else if (!strcmp(argv[i], "--uniform")) {
            refineAdaptive = false;
        } else if (!strcmp(argv[i], "--fvar")) {
            refineFVar = true;
        } else if (!strcmp(argv[i], "--normals")) {
            computeNormals = true;
        } else if (!strcmp(argv[i], "--maxlevel") && (i+1) < argc) {
            maxlevel = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--tesslevel") && (i+1) < argc) {
            tesslevel = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--shape") && (i+1) < argc) {
            filepath = argv[++i];
        } else {
            printf("Parameters :\n");
            printf("  --adaptive         : use feature adaptive refinement.\n");
            printf("  --uniform          : use uniform refinement.\n");
            printf("  --normals          : compute adaptive normals.\n");
            printf("  --fvar             : compute adaptive fvar values.\n");
            printf("  --maxlevel <int>   : topological refinement level.\n");
            printf("  --tesslevel <int>  : geometric tessellation level.\n");
            printf("  --shape <filepath> : read .obj file.\n");

            return EXIT_FAILURE;
        }
    }

    //
    // 1) Load and refine the mesh topology
    //

    // Create a TopologyRefiner
    Far::TopologyRefiner * refiner =
        filepath.empty() ? createDefaultTopologyRefiner()
                         : createTopologyRefiner(filepath);

    // Uniformly/Adaptively refine the topology up to 'maxlevel'
    if (refineAdaptive) {
        refiner->RefineAdaptive(
                Far::TopologyRefiner::AdaptiveOptions(maxlevel));
    } else {
        refiner->RefineUniform(
                Far::TopologyRefiner::UniformOptions(maxlevel));
    }

    //
    // 2) Create a patch table for the refined topology
    //

    // FaceVarying topology for UV data
    refineFVar = refineFVar && (refiner->GetNumFVarChannels() > 0);
    const int numFVarChannels = 1;
    const int uvChannel = 0;

    // Create PatchTable
    Far::PatchTable const * patchTable = NULL;
    {
        Far::PatchTableFactory::Options options;
        options.SetEndCapType(
            Far::PatchTableFactory::Options::ENDCAP_BSPLINE_BASIS);

        // Include face-varying topology
        if (refineAdaptive and refineFVar) {
            options.generateFVarTables = true;
            options.numFVarChannels = numFVarChannels;
            options.fvarChannelIndices = &uvChannel;
        }

        patchTable = Far::PatchTableFactory::Create(*refiner, options);
    }

    //
    // 3) Create stencil tables to refine primvar values
    //

    // Create Vertex StencilTable
    Far::StencilTable const * stencilTable = NULL;
    {
        Far::StencilTableFactory::Options options;
        options.interpolationMode =
                Far::StencilTableFactory::INTERPOLATE_VERTEX;
        options.generateOffsets = true;
        options.generateIntermediateLevels = refineAdaptive;

        stencilTable = Far::StencilTableFactory::Create(*refiner, options);

        // Append local point stencils
        if (Far::StencilTable const * localPointStencilTable =
            patchTable->GetLocalPointStencilTable()) {
            if (Far::StencilTable const * combinedTable =
                Far::StencilTableFactory::AppendLocalPointStencilTable(
                    *refiner, stencilTable,
                    localPointStencilTable)) {
                delete stencilTable;
                stencilTable = combinedTable;
            }
        }
    }

    // Create FaceVarying StencilTable
    Far::StencilTable const * stencilTableFVar = NULL;
    if (refineAdaptive && refineFVar) {
        Far::StencilTableFactory::Options options;
        options.interpolationMode =
                Far::StencilTableFactory::INTERPOLATE_FACE_VARYING;
        options.fvarChannel = uvChannel;
        options.generateOffsets = true;
        options.generateIntermediateLevels = refineAdaptive;

        stencilTableFVar = Far::StencilTableFactory::Create(*refiner, options);

        // Append local point stencils
        if (Far::StencilTable const * localPointFaceVaryingStencilTable =
            patchTable->GetLocalPointFaceVaryingStencilTable()) {
            if (Far::StencilTable const * combinedTable =
                Far::StencilTableFactory::AppendLocalPointStencilTable(
                    *refiner, stencilTableFVar,
                    localPointFaceVaryingStencilTable)) {
                delete stencilTableFVar;
                stencilTableFVar = combinedTable;
            }
        }
    }

    //
    // 4) Tessellate the topology
    //

    Tessellator const * tess = NULL;
    int nEvalPoints = 0;
    if (refineAdaptive) {
        tess = new Tessellator(refiner, patchTable, tesslevel);
        nEvalPoints = tess->GetNumPatchCoords();
    }

    //
    // 5) Determine the layout of primvar data buffers
    //

    int nBasePoints = refiner->GetLevel(0).GetNumVertices();
    int nTotalPoints = stencilTable->GetNumControlVertices() +
                       stencilTable->GetNumStencils();

    int nBaseUVs = 0;
    int nTotalUVs = 0;
    if (refineAdaptive && refineFVar) {
        nBaseUVs = refiner->GetLevel(0).GetNumFVarValues(uvChannel);
        nTotalUVs = stencilTableFVar->GetNumControlVertices() +
                    stencilTableFVar->GetNumStencils();
    }

    // Buffer layouts: eval points only
    Osd::BufferDescriptor evalPointsOnlyDesc(0, 3, 3);
    Osd::BufferDescriptor evalPointsDuDvDesc(0, 3, 9);

    // Buffer layouts: eval points interleaved with derivatives
    Osd::BufferDescriptor evalDuDesc(3, 3, 9);
    Osd::BufferDescriptor evalDvDesc(6, 3, 9);

    // Buffer layouts: refined points
    Osd::BufferDescriptor basePointsDesc(0, 3, 3);
    Osd::BufferDescriptor refinedPointsDesc(
        nBasePoints*basePointsDesc.stride+basePointsDesc.offset, 3, 3);

    // Buffer layouts: eval UV
    Osd::BufferDescriptor evalUVDesc(0, 2, 2);

    // Buffer layouts: refined UV
    Osd::BufferDescriptor baseUVDesc(0, 2, 2);
    Osd::BufferDescriptor refinedUVDesc(
        nBaseUVs*baseUVDesc.stride+baseUVDesc.offset, 2, 2);

    // Buffer for points primvar
    Osd::CpuVertexBuffer * pointsBuffer =
        Osd::CpuVertexBuffer::Create(3, nTotalPoints);

    // Buffer for UV primvar
    Osd::CpuVertexBuffer * uvBuffer = NULL;
    if (refineAdaptive && refineFVar) {
        uvBuffer = Osd::CpuVertexBuffer::Create(2, nTotalUVs);
    }

    // Buffers and descriptors for patch evaluation
    Osd::BufferDescriptor & evalPointsDesc =
        computeNormals ? evalPointsDuDvDesc : evalPointsOnlyDesc;

    Osd::CpuVertexBuffer * evalPointsBuffer = NULL;
    Osd::CpuVertexBuffer * evalUVBuffer = NULL;

    Osd::CpuVertexBuffer * patchCoords = NULL;
    Osd::CpuPatchTable * evalPatchTable = NULL;

    if (refineAdaptive) {
        evalPointsBuffer =
            Osd::CpuVertexBuffer::Create(evalPointsDesc.stride, nEvalPoints);

        if (refineFVar) {
            evalUVBuffer =
                Osd::CpuVertexBuffer::Create(evalUVDesc.stride, nEvalPoints);
        }

        patchCoords = Osd::CpuVertexBuffer::Create(5, nEvalPoints);
        patchCoords->UpdateData((float*)tess->GetPatchCoords(), 0, nEvalPoints);

        evalPatchTable = Osd::CpuPatchTable::Create(patchTable);
    }

    //
    // 6) Refine and evaluate primvar data
    //

    // Pack the control vertex data at the start of the points buffer
    pointsBuffer->UpdateData(&basePoints[0], 0, nBasePoints);

    // Refine vertex points (basePoints -> refinedPoints)
    Osd::CpuEvaluator::EvalStencils(
            pointsBuffer, basePointsDesc,
            pointsBuffer, refinedPointsDesc,
            stencilTable);

    // Evaluate positions and derivatives
    if (refineAdaptive && computeNormals) {
        Osd::CpuEvaluator::EvalPatches(
            pointsBuffer, basePointsDesc,
            evalPointsBuffer, evalPointsDuDvDesc,
            evalPointsBuffer, evalDuDesc,
            evalPointsBuffer, evalDvDesc,
            nEvalPoints, patchCoords,
            evalPatchTable);

    } else if (refineAdaptive) {
        Osd::CpuEvaluator::EvalPatches(
            pointsBuffer, basePointsDesc,
            evalPointsBuffer, evalPointsOnlyDesc,
            nEvalPoints, patchCoords,
            evalPatchTable);
    }

    // Evaluate FaceVarying values at patch points
    if (refineAdaptive && refineFVar) {
        // Pack the UV values at the start of the uv buffer
        uvBuffer->UpdateData(&baseUVs[0], 0, nBaseUVs);

        // Refine UV values (basePoints -> refinedPoints)
        Osd::CpuEvaluator::EvalStencils(
            uvBuffer, baseUVDesc,
            uvBuffer, refinedUVDesc,
            stencilTableFVar);

        // Evaluate UV values at patch points
        Osd::CpuEvaluator::EvalPatchesFaceVarying(
            uvBuffer, baseUVDesc,
            evalUVBuffer, evalUVDesc,
            nEvalPoints, patchCoords,
            evalPatchTable, uvChannel);
    }

    //
    // 7) Emit resulting mesh data
    //

    if (refineAdaptive) {
        printf("# %s adaptive tessellation maxlevel=%d tesslevel=%d\n",
               modelName.c_str(), maxlevel, tesslevel);

        float const * evalPoints = evalPointsBuffer->BindCpuBuffer();
        for (int i=0; i<nEvalPoints; ++i) {
            float const * point = evalPoints + evalPointsDesc.stride*i;
            printf("v %f %f %f\n", point[0], point[1], point[2]);
        }

        if (computeNormals) {
            for (int i=0; i<nEvalPoints; ++i) {
                float const * du = evalPoints
                                 + evalDuDesc.stride*i + evalDuDesc.offset;
                float const * dv = evalPoints
                                 + evalDvDesc.stride*i + evalDvDesc.offset;
                float normal[3];
                computeNormal(normal, du, dv);
                printf("vn %f %f %f\n", normal[0], normal[1], normal[2]);
            }
        }

        float const * evalUV = NULL;
        if (refineAdaptive && refineFVar) {
            evalUV = evalUVBuffer->BindCpuBuffer();
        }

        int const * faceCounts = tess->GetFaceVertexCounts();
        for (int i=0, v=0, fv=0; i<tess->GetNumFaceVertexCounts(); ++i) {
            int const * faceVerts = tess->GetFaceVertexIndices();
            int faceVaryingVert = fv+1;
            if (refineFVar) {
                for (int j=0; j<faceCounts[i]; ++j) {
                    float const * uv = evalUV
                                     + evalUVDesc.stride*(faceVerts[v+j]);
                    printf("vt %f %f\n", uv[0], uv[1]);
                    ++fv;
                }
            }
            printf("f");
            for (int j=0; j<faceCounts[i]; ++j) {
                int faceVert = faceVerts[v++]+1;
                if (computeNormals && refineFVar) {
                    printf(" %d/%d/%d", faceVert, faceVaryingVert+j, faceVert);
                } else if (refineFVar) {
                    printf(" %d/%d", faceVert, faceVaryingVert+j);
                } else if (computeNormals) {
                    printf(" %d//%d", faceVert, faceVert);
                } else {
                    printf(" %d", faceVert);
            }
            }
            printf("\n");
        }

    } else {
        printf("# %s uniform refinement maxlevel=%d\n",
               modelName.c_str(), maxlevel);

        int nRefinedPoints = refiner->GetLevel(maxlevel).GetNumVertices();

        float const * refinedPoints =
            pointsBuffer->BindCpuBuffer() + refinedPointsDesc.offset;
        for (int i=0; i<nRefinedPoints; ++i) {
            float const * point = refinedPoints + 3*i;
            printf("v %f %f %f\n", point[0], point[1], point[2]);
        }

        for (int array=0; array<patchTable->GetNumPatchArrays(); ++array) {
            for (int patch=0; patch<patchTable->GetNumPatches(array); ++patch) {
                Far::ConstIndexArray faceVerts =
                    patchTable->GetPatchVertices(array, patch);
                printf("f");
                for (int v=0; v<faceVerts.size(); ++v) {
                    printf(" %d", faceVerts[v]+1-nBasePoints);
                }
                printf("\n");
            }
        }
    }

    delete refiner;

    delete stencilTable;
    delete patchTable;
    delete pointsBuffer;

    delete tess;
    delete patchCoords;
    delete evalPointsBuffer;
    delete evalPatchTable;
}

//------------------------------------------------------------------------------
