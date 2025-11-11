//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../bfr/surfaceData.h"
#include "../bfr/patchTree.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {
namespace internal {

//
//  Constructors and other methods to manage data members for copy and
//  destruction:
//
SurfaceData::SurfaceData() : _cvIndices(), _param(),
    _isValid(false),
    _isDouble(false),
    _isRegular(true),
    _isLinear(false),
    _regPatchType(0),
    _regPatchMask(0),
    _irregPatch() {
}

SurfaceData & 
SurfaceData::operator=(SurfaceData const & src) {

    //  No need to explicitly manage pre-existing resources in destination
    //  as they will be either re-used or released when re-assigned

    //  No copy/operator= supported by StackBuffer so resize and copy:
    _cvIndices.SetSize(src._cvIndices.GetSize());
    std::memcpy(&_cvIndices[0],
        &src._cvIndices[0], src._cvIndices.GetSize() * sizeof(Index));

    _param = src._param;

    _isValid      = src._isValid;
    _isDouble     = src._isDouble;
    _isRegular    = src._isRegular;
    _isLinear     = src._isLinear;
    _regPatchType = src._regPatchType;
    _regPatchMask = src._regPatchMask;
    _irregPatch   = src._irregPatch;

    return *this;
}

void
SurfaceData::invalidate() {

    //  Release any attached memory before marking as invalid:
    _irregPatch = 0;

    _isValid = false;
}

} // end namespace internal
} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv
