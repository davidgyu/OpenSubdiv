//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_COMMON_PATCH_COLORS_H
#define OPENSUBDIV_EXAMPLES_COMMON_PATCH_COLORS_H

#include <opensubdiv/far/patchTable.h>

// returns a unique color for each type of feature-adaptive patches
float const * getAdaptivePatchColor(OpenSubdiv::Far::PatchDescriptor const & desc);


#endif /* OPENSUBDIV_EXAMPLES_COMMON_PATCH_COLORS_H */
