//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_D3D11_PTEX_MIPMAP_TEXTURE_H
#define OPENSUBDIV_EXAMPLES_D3D11_PTEX_MIPMAP_TEXTURE_H

#include <opensubdiv/osd/nonCopyable.h>

#include <Ptexture.h>

struct ID3D11Buffer;
struct ID3D11Texture2D;
struct ID3D11DeviceContext;
struct ID3D11ShaderResourceView;

class D3D11PtexMipmapTexture : OpenSubdiv::Osd::NonCopyable<D3D11PtexMipmapTexture> {
public:
    static D3D11PtexMipmapTexture * Create(ID3D11DeviceContext *deviceContext,
                                              PtexTexture * reader,
                                              int maxLevels=10,
                                              size_t targetMemory=0);

    /// Returns GLSL shader snippet to fetch ptex
    static const char *GetShaderSource();

    /// Returns the texture buffer containing the layout of the ptex faces
    /// in the texels texture array.
    ID3D11Buffer *GetLayoutTextureBuffer() const { return _layout; }

    ID3D11ShaderResourceView **GetLayoutSRV() { return &_layoutSRV; }

    /// Returns the texels texture array.
    ID3D11Texture2D *GetTexelsTexture() const { return _texels; }

    ID3D11ShaderResourceView **GetTexelsSRV() { return &_texelsSRV; }

    /// Returns the amount of allocated memory (in byte)
    size_t GetMemoryUsage() const { return _memoryUsage; }

    ~D3D11PtexMipmapTexture();

private:
    D3D11PtexMipmapTexture();

    int _width,   // width / height / depth of the 3D texel buffer
        _height,
        _depth;

    int _format;  // texel color format

    ID3D11Buffer *_layout;     // per-face lookup table
    ID3D11Texture2D *_texels;  // texel data
    ID3D11ShaderResourceView *_layoutSRV;
    ID3D11ShaderResourceView *_texelsSRV;

    size_t _memoryUsage;  // total amount of memory used (estimate)
};

#endif  // OPENSUBDIV_EXAMPLES_D3D11_PTEX_TEXTURE_H
