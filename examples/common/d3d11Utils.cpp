//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "d3d11Utils.h"
#include <D3DCompiler.h>

namespace D3D11Utils {

ID3DBlob *
CompileShader(const char *src, const char *entry, const char *spec) {
    DWORD dwShaderFlags = D3DCOMPILE_ENABLE_STRICTNESS;
#if defined(DEBUG) || defined(_DEBUG)
      dwShaderFlags |= D3DCOMPILE_DEBUG;
#endif

    ID3DBlob *pErrorBlob;
    ID3DBlob *pBlob;
    HRESULT hr = D3DCompile(src, strlen(src),
                            NULL,NULL,NULL, entry, spec,
                            dwShaderFlags, 0, &pBlob, &pErrorBlob);
    if (FAILED(hr)) {
        if (pErrorBlob) {
            OutputDebugStringA((char*)pErrorBlob->GetBufferPointer());
            pErrorBlob->Release();
        }
        return NULL;
    }
    if (pErrorBlob)
        pErrorBlob->Release();
    return pBlob;
}

}   // D3D11Utils
