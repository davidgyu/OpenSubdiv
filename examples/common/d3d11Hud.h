//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_D3D11_HUD_H
#define OPENSUBDIV_EXAMPLES_D3D11_HUD_H

#include <D3D11.h>
#include "hud.h"

class D3D11hud : public Hud
{
public:
    D3D11hud(ID3D11DeviceContext *deviceContext);
    ~D3D11hud();

    void Init(int width, int height) {
        Init(width, height, width, height);
    }

    void Rebuild(int width, int height) {
        Rebuild(width, height, width, height);
    }

    virtual void Init(int width, int height, int framebufferWidth, int framebufferHeight);

    virtual void Rebuild(int width, int height,
                         int framebufferWidth, int framebufferHeight);

    virtual bool Flush();

private:
    ID3D11DeviceContext *_deviceContext;
    ID3D11Buffer *_vbo;
    ID3D11Buffer *_staticVbo;
    ID3D11Texture2D *_fontTexture;
    ID3D11InputLayout *_inputLayout;
    ID3D11ShaderResourceView *_shaderResourceView;
    ID3D11SamplerState *_samplerState;
    ID3D11VertexShader *_vertexShader;
    ID3D11PixelShader *_pixelShader;
    ID3D11RasterizerState *_rasterizerState;
    ID3D11Buffer* _constantBuffer;
    int _staticVboCount;
};

#endif  // OPENSUBDIV_EXAMPLES_D3D11_HUD_H
