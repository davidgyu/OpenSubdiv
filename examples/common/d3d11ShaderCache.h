//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_D3D11_SHADER_CACHE_H
#define OPENSUBDIV_EXAMPLES_D3D11_SHADER_CACHE_H

#include <string>
#include "./shaderCache.h"

struct ID3D11VertexShader;
struct ID3D11HullShader;
struct ID3D11DomainShader;
struct ID3D11GeometryShader;
struct ID3D11PixelShader;
struct ID3D11Device;
struct ID3D11InputLayout;
struct D3D11_INPUT_ELEMENT_DESC;

class D3D11DrawConfig {
public:
    D3D11DrawConfig();
    ~D3D11DrawConfig();

    bool CompileVertexShader(const std::string &target,
                             const std::string &entry,
                             const std::string &source,
                             ID3D11InputLayout ** ppInputLayout,
                             D3D11_INPUT_ELEMENT_DESC const *pInputElementDescs,
                             int numInputElements,
                             ID3D11Device * pd3dDevice);
    bool CompileHullShader(const std::string &target,
                           const std::string &entry,
                           const std::string &source,
                           ID3D11Device * pd3dDevice);
    bool CompileDomainShader(const std::string &target,
                             const std::string &entry,
                             const std::string &source,
                             ID3D11Device * pd3dDevice);
    bool CompileGeometryShader(const std::string &target,
                               const std::string &entry,
                               const std::string &source,
                               ID3D11Device * pd3dDevice);
    bool CompilePixelShader(const std::string &target,
                            const std::string &entry,
                            const std::string &source,
                            ID3D11Device * pd3dDevice);

    ID3D11VertexShader   *GetVertexShader() const { return _vertexShader; }
    ID3D11HullShader     *GetHullShader() const { return _hullShader; }
    ID3D11DomainShader   *GetDomainShader() const { return _domainShader; }
    ID3D11GeometryShader *GetGeometryShader() const { return _geometryShader; }
    ID3D11PixelShader    *GetPixelShader() const { return _pixelShader; }

private:
    ID3D11VertexShader   *_vertexShader;
    ID3D11HullShader     *_hullShader;
    ID3D11DomainShader   *_domainShader;
    ID3D11GeometryShader *_geometryShader;
    ID3D11PixelShader    *_pixelShader;
};

// workaround for template alias
#if 0
template <typename DESC_TYPE>
using D3D11ShaderCache = ShaderCacheT<DESC_TYPE, D3D11DrawConfig>;
#else
template <typename DESC_TYPE>
class D3D11ShaderCache : public ShaderCacheT<DESC_TYPE, D3D11DrawConfig> {
};
#endif

#endif  // OPENSUBDIV_EXAMPLES_D3D11_SHADER_CACHE_H
