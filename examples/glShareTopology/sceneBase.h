//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_SCENE_BASE_H
#define OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_SCENE_BASE_H

#include "glLoader.h"

#include <opensubdiv/far/patchDescriptor.h>
#include <opensubdiv/far/patchTable.h>
#include <opensubdiv/osd/bufferDescriptor.h>

struct Shape;

class SceneBase {
public:
    enum EndCap      { kEndCapBSplineBasis,
                       kEndCapGregoryBasis };

    struct Options {
        Options() : adaptive(true), endCap(kEndCapGregoryBasis) { }

        bool adaptive;
        int endCap;
    };

    struct Object {
        int topologyIndex;
        int vertsOffset;        // an offset within the VBO
    };

    struct PatchArray {
        OpenSubdiv::Far::PatchDescriptor desc;
        int numPatches;
        int indexOffset;        // an offset within the index buffer
        int primitiveIDOffset;  // an offset within the patch param buffer
    };
    typedef std::vector<PatchArray> PatchArrayVector;

    struct Topology {
        int numVerts;
        PatchArrayVector patchArrays;
        std::vector<float> restPosition;
    };

    struct Batch {
        OpenSubdiv::Far::PatchDescriptor desc;
        int count;
        int stride;
        GLuint dispatchBuffer;
    };
    typedef std::vector<Batch> BatchVector;

    /// Constructor.
    SceneBase(Options const &options);

    /// Destructor.
    virtual ~SceneBase();

    /// trivial accessors
    int GetNumObjects() const { return (int)_objects.size(); }

    PatchArrayVector const &GetPatchArrays(int object) const {
        return _topologies[_objects[object].topologyIndex].patchArrays;
    }

    BatchVector const &GetBatches() {
        if (_batches.empty()) buildBatches();
        return _batches;
    }

    int GetVertsOffset(int object) const {
        return _objects[object].vertsOffset;
    }

    std::vector<float> const &GetRestPosition(int object) const {
        return _topologies[_objects[object].topologyIndex].restPosition;
    }

    GLuint GetPatchParamTexture() const {
        return _patchParamTexture;
    }

    GLuint GetIndexBuffer() const {
        return _indexBuffer;
    }

    // allocate batched vbo
    virtual size_t AllocateVBO(int numVerts,
                               OpenSubdiv::Osd::BufferDescriptor const &vertexDesc,
                               OpenSubdiv::Osd::BufferDescriptor const &varyingDesc,
                               bool interleaved) = 0;

    // refine an object
    virtual void Refine(int object) =0;

    virtual void Synchronize() = 0;

    virtual void UpdateVertexBuffer(int vertsOffset, std::vector<float> const &src)=0;

    virtual void UpdateVaryingBuffer(int vertsOffset, std::vector<float> const &src)=0;

    virtual GLuint BindVertexBuffer() = 0;

    virtual GLuint BindVaryingBuffer() = 0;

    // build batched index buffer and patchparam texture buffer
    size_t CreateIndexBuffer();

    void AddTopology(Shape const *shape, int level, bool varying);

    int AddObjects(int numObjects);

    size_t GetStencilTableSize() const { return _stencilTableSize; }

protected:
    int createStencilTable(Shape const *shape, int level, bool varying,
                           OpenSubdiv::Far::PatchTable const **patchTableOut);

    void buildBatches();

    virtual size_t createMeshRefiner(
        OpenSubdiv::Far::StencilTable const * vertexStencils,
        OpenSubdiv::Far::StencilTable const * varyingStencils,
        int numControlVertices) = 0;

    Options _options;

    std::vector<Object> _objects;
    std::vector<Topology> _topologies;
    std::vector<OpenSubdiv::Far::PatchTable const *> _patchTables;
    GLuint _indexBuffer;
    GLuint _patchParamTexture;
    BatchVector _batches;
    size_t _stencilTableSize;

};

#endif  // OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_SCENE_BASE_H
