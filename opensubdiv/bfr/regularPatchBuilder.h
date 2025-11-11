//
//   Copyright 2021 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_REGULAR_PATCH_BUILDER_H
#define OPENSUBDIV3_REGULAR_PATCH_BUILDER_H

#include "../version.h"

#include "../bfr/faceSurface.h"
#include "../far/patchDescriptor.h"

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Bfr {

//
//  RegularPatchBuilder ...
//
class RegularPatchBuilder {
public:
    typedef FaceSurface::Index Index;

public:
    RegularPatchBuilder(FaceSurface const & surfaceDescription);
    ~RegularPatchBuilder() { }

    //  Debugging...
    void print(Index const cvIndices[] = 0) const;

public:
    //  Methods to query the number and indices of control vertices:
    int GetNumControlVertices() const { return _patchSize; }

    int GatherControlVertexIndices(Index cvIndices[]) const;

public:
    //  Methods to query patch properties:
    bool IsQuadPatch() const { return _isQuad; }
    bool IsBoundaryPatch() const { return _isBoundary; }

    Far::PatchDescriptor::Type GetPatchType() const { return _patchType; }

    //  Note the bit-mask here is specific for use with Far::PatchParam
    int GetPatchParamBoundaryMask() const { return _boundaryMask; }

public:
    //  Static methods for use without a FaceSurface:
    static int GetPatchSize(int regFaceSize) {
        return (regFaceSize == 4) ? 16 : 12;
    }
    static Far::PatchDescriptor::Type GetPatchType(int regFaceSize) {
        return (regFaceSize == 4) ? Far::PatchDescriptor::REGULAR :
                                    Far::PatchDescriptor::LOOP;
    }
    static int GetBoundaryMask(int regFaceSize, Index const patchPoints[]);

private:
    //  Internal methods for assembling quad and tri patches:
    void gatherInteriorPatchPoints4(Index cvIndices[]) const;
    void gatherBoundaryPatchPoints4(Index cvIndices[]) const;

    void gatherInteriorPatchPoints3(Index cvIndices[]) const;
    void gatherBoundaryPatchPoints3(Index cvIndices[]) const;

private:
    //  Private members:
    FaceSurface const & _surface;

    unsigned int _isQuad     : 1;
    unsigned int _isBoundary : 1;
    int          _boundaryMask;
    int          _patchSize;

    Far::PatchDescriptor::Type _patchType;
};

} // end namespace Bfr

} // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

} // end namespace OpenSubdiv

#endif /* OPENSUBDIV3_REGULAR_PATCH_BUILDER_H */
