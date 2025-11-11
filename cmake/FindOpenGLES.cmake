#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

# - Try to find OpenGLES
# Once done this will define
#  
#  OPENGLES_FOUND        - system has OpenGLES
#  OPENGLES_INCLUDE_DIR  - the GL include directory
#  OPENGLES_LIBRARIES    - Link these to use OpenGLES

if(ANDROID)
    FIND_PATH( OPENGLES_INCLUDE_DIR
        GLES2/gl2.h
        "${ANDROID_STANDALONE_TOOLCHAIN}/usr/include"
    )

    FIND_LIBRARY( OPENGLES_LIBRARIES
        NAMES
            GLESv2
        PATHS
            "${ANDROID_STANDALONE_TOOLCHAIN}/usr/lib"
    )

elseif(IOS)
    FIND_PATH( OPENGLES_INCLUDE_DIR
        OpenGLES/ES2/gl.h
    )

    FIND_LIBRARY( OPENGLES_FRAMEWORKS OpenGLES )

    if(OPENGLES_FRAMEWORKS)
        set( OPENGLES_LIBRARIES "-framework OpenGLES" )
    endif()

endif()

SET( OPENGLES_FOUND "NO" )
IF(OPENGLES_LIBRARIES)
    SET( OPENGLES_FOUND "YES" )
ENDIF(OPENGLES_LIBRARIES)

MARK_AS_ADVANCED(
  OPENGLES_INCLUDE_DIR
  OPENGLES_LIBRARIES
)

