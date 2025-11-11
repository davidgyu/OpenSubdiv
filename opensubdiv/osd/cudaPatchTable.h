//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_CUDA_PATCH_TABLE_H
#define OPENSUBDIV3_OSD_CUDA_PATCH_TABLE_H

#include "../version.h"

#include "../osd/nonCopyable.h"
#include "../osd/types.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far{
    class PatchTable;
};

namespace Osd {

/// \brief CUDA patch table
///
/// This class is a cuda buffer representation of Far::PatchTable.
///
/// CudaEvaluator consumes this table to evaluate on the patches.
///
///
class CudaPatchTable : private NonCopyable<CudaPatchTable> {
public:
    /// Creator. Returns NULL if error
    static CudaPatchTable *Create(Far::PatchTable const *patchTable,
                                  void *deviceContext = NULL);
    /// Destructor
    ~CudaPatchTable();

    /// Returns the cuda memory of the array of Osd::PatchArray buffer
    void *GetPatchArrayBuffer() const { return _patchArrays; }

    /// Returns the cuda memory of the patch control vertices
    void *GetPatchIndexBuffer() const { return _indexBuffer; }

    /// Returns the cuda memory of the array of Osd::PatchParam buffer
    void *GetPatchParamBuffer() const { return _patchParamBuffer; }

    /// Returns the cuda memory of the array of Osd::PatchArray buffer
    void *GetVaryingPatchArrayBuffer() const {
        return _varyingPatchArrays;
    }
    /// Returns the cuda memory of the array of varying control vertices
    void *GetVaryingPatchIndexBuffer() const {
        return _varyingIndexBuffer;
    }

    /// Returns the number of face-varying channels buffers
    int GetNumFVarChannels() const { return (int)_fvarPatchArrays.size(); }

    /// Returns the cuda memory of the array of Osd::PatchArray buffer
    void *GetFVarPatchArrayBuffer(int fvarChannel) const {
        return _fvarPatchArrays[fvarChannel];
    }

    /// Returns the cuda memory of the array of face-varying control vertices
    void *GetFVarPatchIndexBuffer(int fvarChannel = 0) const {
        return _fvarIndexBuffers[fvarChannel];
    }

    /// Returns the cuda memory of the array of face-varying param
    void *GetFVarPatchParamBuffer(int fvarChannel = 0) const {
        return _fvarParamBuffers[fvarChannel];
    }

protected:
    CudaPatchTable();

    bool allocate(Far::PatchTable const *patchTable);

    void *_patchArrays;
    void *_indexBuffer;
    void *_patchParamBuffer;

    void *_varyingPatchArrays;
    void *_varyingIndexBuffer;

    std::vector<void *> _fvarPatchArrays;
    std::vector<void *> _fvarIndexBuffers;
    std::vector<void *> _fvarParamBuffers;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_CUDA_PATCH_TABLE_H
