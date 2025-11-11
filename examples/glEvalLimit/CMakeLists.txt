#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

# *** glEvalLimit ***

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
    "${OPENGL_LOADER_INCLUDE_DIRS}"
    "${GLFW_INCLUDE_DIR}"
)

list(APPEND PLATFORM_LIBRARIES
    "${OSD_LINK_TARGET}"
    "${OPENGL_LOADER_LIBRARIES}"
    "${GLFW_LIBRARIES}"
)

if (OPENCL_FOUND)
    include_directories("${OPENCL_INCLUDE_DIRS}")
    list(APPEND PLATFORM_LIBRARIES OpenGL::GL)
endif()

if( TBB_FOUND )
    list(APPEND PLATFORM_LIBRARIES
        TBB::tbb
    )
endif()

osd_add_glfw_executable(glEvalLimit "examples"
    glEvalLimit.cpp
    particles.cpp
    particles.h
    "${INC_FILES}"
    $<TARGET_OBJECTS:regression_common_obj>
    $<TARGET_OBJECTS:regression_far_utils_obj>
    $<TARGET_OBJECTS:examples_common_gl_obj>
)

target_link_libraries(glEvalLimit
    ${PLATFORM_LIBRARIES}
)

install(TARGETS glEvalLimit DESTINATION "${CMAKE_BINDIR_BASE}")
