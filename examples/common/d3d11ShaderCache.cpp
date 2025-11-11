//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "d3d11ShaderCache.h"

#include <D3D11.h>
#include <D3Dcompiler.h>

#include <opensubdiv/far/error.h>

D3D11DrawConfig::D3D11DrawConfig()
    : _vertexShader(NULL), _hullShader(NULL), _domainShader(NULL),
      _geometryShader(NULL), _pixelShader(NULL) {
}

D3D11DrawConfig::~D3D11DrawConfig() {
    if (_vertexShader)   _vertexShader->Release();
    if (_hullShader)     _hullShader->Release();
    if (_domainShader)   _domainShader->Release();
    if (_geometryShader) _geometryShader->Release();
    if (_pixelShader)    _pixelShader->Release();
}

static ID3DBlob *
_CompileShader(std::string const &target,
               std::string const &entry,
               std::string const &shaderSource) {
    DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#ifdef _DEBUG
    dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

    ID3DBlob* pBlob = NULL;
    ID3DBlob* pBlobError = NULL;

    HRESULT hr = D3DCompile(shaderSource.c_str(), shaderSource.size(),
                            NULL, NULL, NULL,
                            entry.c_str(), target.c_str(),
                            dwShaderFlags, 0, &pBlob, &pBlobError);
    if (FAILED(hr)) {
        if ( pBlobError != NULL ) {
            OpenSubdiv::Far::Error(OpenSubdiv::Far::FAR_RUNTIME_ERROR,
                     "Error compiling HLSL shader: %s\n",
                     (CHAR*)pBlobError->GetBufferPointer());
            pBlobError->Release();
            return NULL;
        }
    }

    return pBlob;
}

bool
D3D11DrawConfig::CompileVertexShader(const std::string &target,
                                     const std::string &entry,
                                     const std::string &source,
                                     ID3D11InputLayout ** ppInputLayout,
                                     D3D11_INPUT_ELEMENT_DESC const * pInputElementDescs,
                                     int numInputElements,
                                     ID3D11Device * pd3dDevice) {
    ID3DBlob * pBlob = NULL;
    pBlob = _CompileShader(target, entry, source);

    HRESULT hr = pd3dDevice->CreateVertexShader(pBlob->GetBufferPointer(),
                                                pBlob->GetBufferSize(),
                                                NULL,
                                                &_vertexShader);
    if (FAILED(hr)) {
        return false;
    }

    if (ppInputLayout && !*ppInputLayout) {
        hr = pd3dDevice->CreateInputLayout(
            pInputElementDescs, numInputElements,
            pBlob->GetBufferPointer(),
            pBlob->GetBufferSize(), ppInputLayout);
        if (FAILED(hr)) {
            return false;
        }
    }
    if (pBlob) pBlob->Release();

    return true;
}

bool
D3D11DrawConfig::CompileHullShader(const std::string &target,
                                   const std::string &entry,
                                   const std::string &source,
                                   ID3D11Device * pd3dDevice) {
    ID3DBlob * pBlob = NULL;
    pBlob = _CompileShader(target, entry, source);

    HRESULT hr = pd3dDevice->CreateHullShader(pBlob->GetBufferPointer(),
                                              pBlob->GetBufferSize(),
                                              NULL,
                                              &_hullShader);
    if (FAILED(hr)) {
        return false;
    }
    if (pBlob) pBlob->Release();

    return true;
}

bool
D3D11DrawConfig::CompileDomainShader(const std::string &target,
                                     const std::string &entry,
                                     const std::string &source,
                                     ID3D11Device * pd3dDevice) {
    ID3DBlob * pBlob = NULL;
    pBlob = _CompileShader(target, entry, source);

    HRESULT hr = pd3dDevice->CreateDomainShader(pBlob->GetBufferPointer(),
                                                pBlob->GetBufferSize(),
                                                NULL,
                                                &_domainShader);
    if (FAILED(hr)) {
        return false;
    }
    if (pBlob) pBlob->Release();

    return true;
}

bool
D3D11DrawConfig::CompileGeometryShader(const std::string &target,
                                       const std::string &entry,
                                       const std::string &source,
                                       ID3D11Device * pd3dDevice) {
    ID3DBlob * pBlob = NULL;
    pBlob = _CompileShader(target, entry, source);

    HRESULT hr = pd3dDevice->CreateGeometryShader(pBlob->GetBufferPointer(),
                                                  pBlob->GetBufferSize(),
                                                  NULL,
                                                  &_geometryShader);
    if (FAILED(hr)) {
        return false;
    }
    if (pBlob) pBlob->Release();

    return true;
}

bool
D3D11DrawConfig::CompilePixelShader(const std::string &target,
                                    const std::string &entry,
                                    const std::string &source,
                                    ID3D11Device * pd3dDevice) {
    ID3DBlob * pBlob = NULL;
    pBlob = _CompileShader(target, entry, source);

    HRESULT hr = pd3dDevice->CreatePixelShader(pBlob->GetBufferPointer(),
                                               pBlob->GetBufferSize(),
                                               NULL,
                                               &_pixelShader);
    if (FAILED(hr)) {
        return false;
    }
    if (pBlob) pBlob->Release();

    return true;
}

