//
//   Copyright 2013 Nvidia
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include <D3D11.h>
#include <D3Dcompiler.h>

//
// Draws an environment sphere centered on the camera w/ a texture
//
class Sky {

public:

    // Constructor (Sky does not own the texture asset)
    Sky(ID3D11Device * device, ID3D11Texture2D * environmentMap);

    ~Sky();

    void Draw(ID3D11DeviceContext * deviceContext, float const mvp[16]);

private:

    void initialize(ID3D11Device * device);

private:

    int numIndices;

    ID3D11VertexShader * vertexShader;
    ID3D11PixelShader * pixelShader;
    ID3D11Buffer * shaderConstants;

    ID3D11InputLayout * inputLayout;
    ID3D11RasterizerState * rasterizerState;
    ID3D11DepthStencilState * depthStencilState;

    ID3D11Texture2D * texture;
    ID3D11ShaderResourceView * textureSRV;
    ID3D11SamplerState * textureSS;

    ID3D11Buffer * sphere;
    ID3D11Buffer * sphereIndices;
};

