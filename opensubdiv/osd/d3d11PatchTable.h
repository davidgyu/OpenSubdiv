//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_GL_PATCH_TABLE_H
#define OPENSUBDIV3_OSD_GL_PATCH_TABLE_H

#include "../version.h"

#include <vector>
#include "../far/patchDescriptor.h"
#include "../osd/nonCopyable.h"
#include "../osd/types.h"

struct ID3D11Buffer;
struct ID3D11ShaderResourceView;
struct ID3D11Device;
struct ID3D11DeviceContext;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far{
    class PatchTable;
};

namespace Osd {

class D3D11PatchTable : private NonCopyable<D3D11PatchTable> {
public:
    using VertexBufferBinding = ID3D11Buffer*;  // buffer binding type

    D3D11PatchTable();
    ~D3D11PatchTable();

    template<typename DEVICE_CONTEXT>
    static D3D11PatchTable *Create(Far::PatchTable const *farPatchTable,
                                   DEVICE_CONTEXT context) {
        return Create(farPatchTable, context->GetDeviceContext());
    }

    static D3D11PatchTable *Create(Far::PatchTable const *farPatchTable,
                                   ID3D11DeviceContext *deviceContext);

    PatchArrayVector const &GetPatchArrays() const {
        return _patchArrays;
    }

    /// Returns the index buffer containing the patch control vertices
    ID3D11Buffer* GetPatchIndexBuffer() const {
        return _indexBuffer;
    }

    /// Returns the SRV containing the patch parameter
    ID3D11ShaderResourceView* GetPatchParamSRV() const {
        return _patchParamBufferSRV;
    }

protected:
    // allocate buffers from patchTable
    bool allocate(Far::PatchTable const *farPatchTable,
                  ID3D11DeviceContext *deviceContext);

    PatchArrayVector _patchArrays;

    ID3D11Buffer             *_indexBuffer;
    ID3D11Buffer             *_patchParamBuffer;
    ID3D11ShaderResourceView *_patchParamBufferSRV;
};


}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_GL_PATCH_TABLE_H
