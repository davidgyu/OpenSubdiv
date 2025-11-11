//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef GL_MESH_H
#define GL_MESH_H

#include "../../regression/common/far_utils.h"
#include <opensubdiv/far/patchTable.h>

#include "../common/glUtils.h"

#include <algorithm>

// Wrapper class for drawing Far meshes & components
class GLMesh {

public:

    enum Component {
        COMP_FACE=0,
        COMP_EDGE,
        COMP_VERT,
        COMP_NUM_COMPONENTS
    };

    enum VertColorMode {
        VERTCOLOR_SOLID=0,
        VERTCOLOR_BY_LEVEL,
        VERTCOLOR_BY_SHARPNESS,
        VERTCOLOR_BY_PARENT_TYPE
    };

    enum EdgeColorMode {
        EDGECOLOR_SOLID=0,
        EDGECOLOR_BY_LEVEL,
        EDGECOLOR_BY_SHARPNESS,
        EDGECOLOR_BY_PATCHTYPE
    };

    enum FaceColorMode {
        FACECOLOR_SOLID=0,
        FACECOLOR_BY_PATCHTYPE
    };

    struct Options {
        Options() : vertColorMode(0), edgeColorMode(0), faceColorMode(0) {}
        unsigned int vertColorMode:3,
                     edgeColorMode:3,
                     faceColorMode:3;
    };

    // -----------------------------------------------------
    // Raw topology initialization
    void Initialize(Options options,
        int nverts, int nfaces, int * vertsperface, int * faceverts,
            float const * vertexData);

    // -----------------------------------------------------
    // Far initialization
    typedef OpenSubdiv::Far::TopologyRefiner TopologyRefiner;

    typedef OpenSubdiv::Far::PatchTable PatchTable;

    void Initialize(Options options, TopologyRefiner const & refiner,
        PatchTable const * patchTable, float const * vertexData);

    void InitializeFVar(Options options, TopologyRefiner const & refiner,
        PatchTable const * patchTable, int channel, int tessFactor, float const * fvarData);

    void InitializeDeviceBuffers();

    // -----------------------------------------------------

    GLMesh();

    ~GLMesh();

    void Draw(Component comp, GLuint transformUB, GLuint lightingUB);

    void SetDiffuseColor(float r, float g, float b, float a);

    void SetFaceColor(int face, float r, float g, float b, float a);

private:

           void setSolidColor(float * color);

    static void setColorByLevel(int level, float * color);

    static void setColorBySharpness(float sharpness, float * color);

    void initializeVertexComponentBuffer(float const * vertexData, int nverts);

    void initializeBuffers(Options options, TopologyRefiner const & refiner,
        float const * vertexData);

    void initializeBuffers(Options options, TopologyRefiner const & refiner,
        PatchTable const & patchTable, float const * vertexData);

    void clearBuffers();


    int _numComps[COMP_NUM_COMPONENTS];

    GLuint _VAO[COMP_NUM_COMPONENTS],
           _VBO[COMP_NUM_COMPONENTS],
           _EAO[COMP_NUM_COMPONENTS],
           _TBOfaceColors;

    std::vector<float> _vbo[COMP_NUM_COMPONENTS];
    std::vector<int>   _eao[COMP_NUM_COMPONENTS];

    std::vector<float > _faceColors;

    float _ambientColor[4],
          _diffuseColor[4];
};

#endif // GL_MESH_H
