//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_D3D11_LEGACY_GREGORY_PATCH_TABLE_H
#define OPENSUBDIV3_OSD_D3D11_LEGACY_GREGORY_PATCH_TABLE_H

#include "../version.h"

#include "../far/patchTable.h"
#include "../osd/nonCopyable.h"

struct ID3D11Buffer;
struct ID3D11ShaderResourceView;
struct ID3D11Device;
struct ID3D11DeviceContext;

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

class D3D11LegacyGregoryPatchTable
    : private NonCopyable<D3D11LegacyGregoryPatchTable> {
public:
    ~D3D11LegacyGregoryPatchTable();

    template<typename DEVICE_CONTEXT>
    static D3D11LegacyGregoryPatchTable *Create(
        Far::PatchTable const *farPatchTable, DEVICE_CONTEXT context) {
        return Create(farPatchTable, context->GetDeviceContext());
    }

    static D3D11LegacyGregoryPatchTable *Create(
        Far::PatchTable const *farPatchTable,
        ID3D11DeviceContext *deviceContext);

    void UpdateVertexBuffer(ID3D11Buffer *vbo,
                            int numVertices, int numVertexElements,
                            ID3D11DeviceContext *pd3d11DeviceContext);

    ID3D11ShaderResourceView* GetVertexSRV() const {
        return _vertexSRV;
    }

    ID3D11ShaderResourceView* GetVertexValenceSRV() const {
        return _vertexValenceSRV;
    }

    ID3D11ShaderResourceView* GetQuadOffsetsSRV() const {
        return _quadOffsetsSRV;
    }

    int GetQuadOffsetsBase(Far::PatchDescriptor::Type type) {
        if (type == Far::PatchDescriptor::GREGORY_BOUNDARY) {
            return _quadOffsetsBase[1];
        }
        return _quadOffsetsBase[0];
    }

protected:
    D3D11LegacyGregoryPatchTable();

private:
    ID3D11Buffer* _vertexValenceBuffer;
    ID3D11Buffer* _quadOffsetsBuffer;
    ID3D11ShaderResourceView* _vertexSRV;
    ID3D11ShaderResourceView* _vertexValenceSRV;
    ID3D11ShaderResourceView* _quadOffsetsSRV;
    int _quadOffsetsBase[2];       // gregory, boundaryGregory
};



}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_D3D11_LEGACY_GREGORY_PATCH_TABLE_H
