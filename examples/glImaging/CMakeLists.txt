#
#   Copyright 2015 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

# *** glImaging ***

set(SHADER_FILES
     shader.glsl
)

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
    "${OPENGL_LOADER_INCLUDE_DIRS}"
    "${GLFW_INCLUDE_DIR}"
)

set(PLATFORM_LIBRARIES
    "${OSD_LINK_TARGET}"
    "${OPENGL_LOADER_LIBRARIES}"
    "${GLFW_LIBRARIES}"
)

if (OPENCL_FOUND)
    include_directories( "${OPENCL_INCLUDE_DIRS}" )
    list(APPEND PLATFORM_LIBRARIES OpenGL::GL)
endif()

osd_stringify("${SHADER_FILES}" INC_FILES)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

osd_add_glfw_executable(glImaging "examples"
    glImaging.cpp
    "${SHADER_FILES}"
    "${INC_FILES}"
    $<TARGET_OBJECTS:regression_common_obj>
    $<TARGET_OBJECTS:examples_common_gl_obj>
)

target_link_libraries(glImaging
    ${PLATFORM_LIBRARIES}
)

install(TARGETS glImaging DESTINATION "${CMAKE_BINDIR_BASE}")

if (NOT NO_GLTESTS)
    add_test(glImaging ${EXECUTABLE_OUTPUT_PATH}/glImaging -w test -l 3 -s 256 256 -a)
endif()

