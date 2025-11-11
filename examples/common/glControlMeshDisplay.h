//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_GL_CONTROL_MESH_DISPLAY_H
#define OPENSUBDIV_EXAMPLES_GL_CONTROL_MESH_DISPLAY_H

#include "glLoader.h"

#include <opensubdiv/far/topologyLevel.h>

class GLControlMeshDisplay {
public:
    GLControlMeshDisplay();
    ~GLControlMeshDisplay();

    void Draw(GLuint pointsVBO, GLint stride,
              const float *modelViewProjectionMatrix);

    void SetTopology(OpenSubdiv::Far::TopologyLevel const &level);

    bool GetEdgesDisplay() const { return _displayEdges; }
    void SetEdgesDisplay(bool display) { _displayEdges = display; }
    bool GetVerticesDisplay() const { return _displayVertices; }
    void SetVerticesDisplay(bool display) { _displayVertices = display; }

private:
    bool createProgram();

    bool _displayEdges;
    bool _displayVertices;

    GLuint _program;
    GLuint _uniformMvpMatrix;
    GLuint _uniformDrawMode;
    GLuint _uniformEdgeSharpness;
    GLuint _attrPosition;
    GLuint _attrVertSharpness;

    GLuint _vao;
    GLuint _vertSharpness;
    GLuint _edgeSharpnessTexture;
    GLuint _edgeIndices;

    int _numEdges, _numPoints;
};

#endif  // OPENSUBDIV_EXAMPLES_GL_CONTROL_MESH_DISPLAY_H
