//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_CL_PATCH_TABLE_H
#define OPENSUBDIV3_OSD_CL_PATCH_TABLE_H

#include "../version.h"

#include "../osd/opencl.h"
#include "../osd/nonCopyable.h"
#include "../osd/types.h"

#include <vector>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Far{
    class PatchTable;
};

namespace Osd {

/// \brief CL patch table
///
/// This class is a CL buffer representation of Far::PatchTable.
///
/// CLEvaluator consumes this table to evaluate on the patches.
///
///
class CLPatchTable : private NonCopyable<CLPatchTable> {
public:
    /// Creator. Returns NULL if error
    static CLPatchTable *Create(Far::PatchTable const *patchTable,
                                cl_context clContext);

    template <typename DEVICE_CONTEXT>
    static CLPatchTable * Create(Far::PatchTable const *patchTable,
                                 DEVICE_CONTEXT context) {
        return Create(patchTable, context->GetContext());
    }

    /// Destructor
    ~CLPatchTable();

    /// Returns the CL memory of the array of Osd::PatchArray buffer
    cl_mem GetPatchArrayBuffer() const { return _patchArrays; }

    /// Returns the CL memory of the patch control vertices
    cl_mem GetPatchIndexBuffer() const { return _indexBuffer; }

    /// Returns the CL memory of the array of Osd::PatchParam buffer
    cl_mem GetPatchParamBuffer() const { return _patchParamBuffer; }

    /// Returns the CL memory of the array of Osd::PatchArray buffer
    cl_mem GetVaryingPatchArrayBuffer() const { return _varyingPatchArrays; }

    /// Returns the CL memory of the varying control vertices
    cl_mem GetVaryingPatchIndexBuffer() const { return _varyingIndexBuffer; }

    /// Returns the number of face-varying channel buffers
    int GetNumFVarChannels() const { return (int)_fvarPatchArrays.size(); }

    /// Returns the CL memory of the array of Osd::PatchArray buffer
    cl_mem GetFVarPatchArrayBuffer(int fvarChannel = 0) const { return _fvarPatchArrays[fvarChannel]; }

    /// Returns the CL memory of the face-varying control vertices
    cl_mem GetFVarPatchIndexBuffer(int fvarChannel = 0) const { return _fvarIndexBuffers[fvarChannel]; }

    /// Returns the CL memory of the array of Osd::PatchParam buffer
    cl_mem GetFVarPatchParamBuffer(int fvarChannel = 0) const { return _fvarParamBuffers[fvarChannel]; }

protected:
    CLPatchTable();

    bool allocate(Far::PatchTable const *patchTable, cl_context clContext);

    cl_mem _patchArrays;
    cl_mem _indexBuffer;
    cl_mem _patchParamBuffer;

    cl_mem _varyingPatchArrays;
    cl_mem _varyingIndexBuffer;

    std::vector<cl_mem> _fvarPatchArrays;
    std::vector<cl_mem> _fvarIndexBuffers;
    std::vector<cl_mem> _fvarParamBuffers;

};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_CL_PATCH_TABLE_H
