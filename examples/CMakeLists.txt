#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

add_subdirectory(common)

if (OPENGL_FOUND AND GLFW_FOUND)

    add_subdirectory(glViewer)
    add_subdirectory(glEvalLimit)
    add_subdirectory(glShareTopology)
    add_subdirectory(glFVarViewer)
    add_subdirectory(glStencilViewer)
    add_subdirectory(glImaging)
    add_subdirectory(farViewer)

    if (OPENGL_4_2_FOUND)
        add_subdirectory(glPaintTest)
    endif()

    if (PTEX_FOUND AND ZLIB_FOUND)
        add_subdirectory(glPtexViewer)
    endif()

endif()

if (DXSDK_FOUND AND NOT NO_DX)

   add_subdirectory(dxViewer)

   if (PTEX_FOUND AND ZLIB_FOUND)
       add_subdirectory(dxPtexViewer)
   endif()

endif()

if (METAL_FOUND AND NOT NO_METAL)

    add_subdirectory(mtlViewer)

    if(PTEX_FOUND)
        add_subdirectory(mtlPtexViewer)
    endif()

endif()
