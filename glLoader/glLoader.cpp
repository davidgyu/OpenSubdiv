//
//   Copyright 2020 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "glLoader.h"

#include <stdio.h>
#include <stdlib.h>


namespace OpenSubdiv {
namespace internal {
namespace GLLoader {


bool
applicationInitializeGL()
{
#if defined(OSD_USES_INTERNAL_GLAPILOADER)
    // -- GLAPILOADER
    return OpenSubdiv::internal::GLApi::glApiLoad();

#elif defined(OSD_USES_GLEW)
    // -- GLEW
#define CORE_PROFILE
#ifdef CORE_PROFILE
    // this is the only way to initialize GLEW (before GLEW 1.13)
    // correctly under core profile context.
    glewExperimental = true;
#endif
    GLenum status = glewInit();
    if (status != GLEW_OK) {
        printf("Failed to initialize glew. Error = %s\n",
               glewGetErrorString(status));
        return false;
    }
#ifdef CORE_PROFILE
    // clear GL errors which were generated during glewInit()
    glGetError();
#endif
#endif
    return true;
}

bool
libraryInitializeGL()
{
#if defined(OSD_USES_INTERNAL_GLAPILOADER)
    return OpenSubdiv::internal::GLApi::glApiLoad();
#else
    // otherwise do nothing
    return true;
#endif
}


}  // namespace GLLoader
}  // namespace internal
}  // namespace OpenSubdiv
