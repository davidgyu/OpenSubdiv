#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

# Try to find GLEW library and include path.
# Once done this will define
#
# GLEW_FOUND
# GLEW_INCLUDE_DIR
# GLEW_LIBRARY
#

include(FindPackageHandleStandardArgs)

if (WIN32)

    find_path(GLEW_INCLUDE_DIR
        NAMES
            GL/glew.h
        HINTS
            "${GLEW_LOCATION}/include"
            "$ENV{GLEW_LOCATION}/include"
        PATHS
            "$ENV{PROGRAMFILES}/GLEW/include"
            "${PROJECT_SOURCE_DIR}/extern/glew/include"
        DOC "The directory where GL/glew.h resides" )

    if ("${CMAKE_GENERATOR}" MATCHES "[Ww]in64" OR
        "${CMAKE_GENERATOR_PLATFORM}" MATCHES "x64")
        set(ARCH x64)
    else()
        set(ARCH x86)
    endif()

    find_library(GLEW_LIBRARY
        NAMES
            glew GLEW glew32s glew32
        HINTS
            "${GLEW_LOCATION}/lib"
            "$ENV{GLEW_LOCATION}/lib"
        PATHS
            "$ENV{PROGRAMFILES}/GLEW/lib"
            "${PROJECT_SOURCE_DIR}/extern/glew/bin"
            "${PROJECT_SOURCE_DIR}/extern/glew/lib"
        PATH_SUFFIXES
            Release/${ARCH}
        DOC "The GLEW library")
endif ()

if (${CMAKE_HOST_UNIX})
    find_path( GLEW_INCLUDE_DIR
        NAMES
            GL/glew.h
        HINTS
            "${GLEW_LOCATION}/include"
            "$ENV{GLEW_LOCATION}/include"
        PATHS
            /usr/include
            /usr/local/include
            /sw/include
            /opt/local/include
            NO_DEFAULT_PATH
            DOC "The directory where GL/glew.h resides"
    )
    find_library( GLEW_LIBRARY
        NAMES
            GLEW glew
        HINTS
            "${GLEW_LOCATION}/lib"
            "$ENV{GLEW_LOCATION}/lib"
        PATHS
            /usr/lib64
            /usr/lib
            /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
            /usr/local/lib64
            /usr/local/lib
            /sw/lib
            /opt/local/lib
            NO_DEFAULT_PATH
            DOC "The GLEW library")
endif ()

find_package_handle_standard_args(GLEW
    REQUIRED_VARS
        GLEW_INCLUDE_DIR
        GLEW_LIBRARY
)
