//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "patchColors.h"

static float _colors[7][4] = {{1.0f,  1.0f,  1.0f,  1.0f},   // regular
                              {1.0f,  0.5f,  0.5f,  1.0f},   // single crease
                              {0.8f,  0.0f,  0.0f,  1.0f},   // boundary
                              {0.0f,  1.0f,  0.0f,  1.0f},   // corner
                              {1.0f,  1.0f,  0.0f,  1.0f},   // gregory
                              {1.0f,  0.5f,  0.0f,  1.0f},   // gregory boundary
                              {1.0f,  0.7f,  0.3f,  1.0f}};  // gregory basis

typedef OpenSubdiv::Far::PatchDescriptor Descriptor;

float const *
getAdaptivePatchColor(Descriptor const & desc) {

    return _colors[(int)(desc.GetType()-Descriptor::REGULAR)];
}

