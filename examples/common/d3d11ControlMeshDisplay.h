//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_D3D11_CONTROL_MESH_DISPLAY_H
#define OPENSUBDIV_EXAMPLES_D3D11_CONTROL_MESH_DISPLAY_H

#include <d3d11.h>
#include <opensubdiv/far/topologyLevel.h>

class D3D11ControlMeshDisplay {
public:
    D3D11ControlMeshDisplay(ID3D11DeviceContext *deviceContext);
    ~D3D11ControlMeshDisplay();

    void Draw(ID3D11Buffer *buffer, int stride,
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

    ID3D11DeviceContext *_deviceContext;
    ID3D11InputLayout *_inputLayout;
    ID3D11VertexShader *_vertexShader;
    ID3D11PixelShader *_pixelShader;
    ID3D11RasterizerState *_rasterizerState;
    ID3D11Buffer *_constantBuffer;
    ID3D11ShaderResourceView *_edgeSharpnessSRV;
    ID3D11Buffer *_edgeSharpness;
    ID3D11Buffer *_edgeIndices;

    int _numEdges, _numPoints;
};

#endif  // OPENSUBDIV_EXAMPLES_D3D11_CONTROL_MESH_DISPLAY_H
