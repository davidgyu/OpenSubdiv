#
#   Copyright 2020 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

if( APPLE OR UNIX )
    add_definitions(-fPIC)
endif()

list(APPEND GLLOADER_SOURCE_FILES
    glLoader.cpp
    glApi.cpp
)

list(APPEND GLLOADER_HEADER_FILES
    glLoader.h
    glApi.h
    khrplatform.h
)

include_directories(
    "${OPENGL_LOADER_INCLUDE_DIRS}"
)

list(APPEND PLATFORM_LIBRARIES
    "${OPENGL_LOADER_LIBRARIES}"
)

add_library(glLoader_obj
    OBJECT
        ${GLLOADER_SOURCE_FILES}
        ${GLLOADER_HEADER_FILES}
)
