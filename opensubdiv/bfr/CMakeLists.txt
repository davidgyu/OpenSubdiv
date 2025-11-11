#
#   Copyright 2021 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# source & headers
set(SOURCE_FILES
     faceSurface.cpp
     faceTopology.cpp
     faceVertex.cpp
     hash.cpp
     irregularPatchBuilder.cpp
     parameterization.cpp
     patchTree.cpp
     patchTreeBuilder.cpp
     refinerSurfaceFactory.cpp
     regularPatchBuilder.cpp
     surface.cpp
     surfaceData.cpp
     surfaceFactory.cpp
     surfaceFactoryCache.cpp
     tessellation.cpp
     vertexDescriptor.cpp
)

set(PRIVATE_HEADER_FILES
     faceSurface.h
     faceTopology.h
     faceVertex.h
     faceVertexSubset.h
     hash.h
     irregularPatchBuilder.h
     patchTree.h
     patchTreeBuilder.h
     pointOperations.h
     regularPatchBuilder.h
     vertexTag.h
)

set(PUBLIC_HEADER_FILES
     irregularPatchType.h
     limits.h
     parameterization.h
     refinerSurfaceFactory.h
     surface.h
     surfaceData.h
     surfaceFactory.h
     surfaceFactoryMeshAdapter.h
     surfaceFactoryCache.h
     tessellation.h
     vertexDescriptor.h
)

set(DOXY_HEADER_FILES ${PUBLIC_HEADER_FILES})

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

if (NOT NO_LIB)

    # Compile objs first for both the CPU and GPU libs -----
    add_library(bfr_obj
        OBJECT
            ${SOURCE_FILES}
            ${PRIVATE_HEADER_FILES}
            ${PUBLIC_HEADER_FILES}
    )

    set_target_properties(bfr_obj
        PROPERTIES
            FOLDER "opensubdiv"
    )

endif()

#-------------------------------------------------------------------------------
osd_add_doxy_headers( "${DOXY_HEADER_FILES}" )

install(
    FILES
        ${PUBLIC_HEADER_FILES}
    DESTINATION
        "${CMAKE_INCDIR_BASE}/bfr"
    PERMISSIONS
        OWNER_READ
        GROUP_READ
        WORLD_READ )

#-------------------------------------------------------------------------------
