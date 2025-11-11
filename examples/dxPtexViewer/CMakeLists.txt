#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

# *** dxViewer ***

set(SHADER_FILES
     shader.hlsl
     skyshader.hlsl
)

set(PLATFORM_LIBRARIES
    "${OSD_LINK_TARGET}"
    "${DXSDK_LIBRARIES}"
    "${PTEX_LIBRARY}"
    "${ZLIB_LIBRARY}"
)

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
    "${DXSDK_INCLUDE_DIR}"
    "${PTEX_INCLUDE_DIR}"
)

if (OPENCL_FOUND)
    include_directories("${OPENCL_INCLUDE_DIRS}")
    list(APPEND PLATFORM_LIBRARIES OpenGL::GL)
endif()

set(SOURCE_FILES
    dxPtexViewer.cpp
    sky.cpp
)

osd_stringify("${SHADER_FILES}" INC_FILES)

include_directories("${CMAKE_CURRENT_BINARY_DIR}")

osd_add_possibly_cuda_executable(dxPtexViewer "examples" WIN32
    "${SOURCE_FILES}"
    "${INC_FILES}"
    $<TARGET_OBJECTS:regression_common_obj>
    $<TARGET_OBJECTS:regression_far_utils_obj>
    $<TARGET_OBJECTS:examples_common_dx11_obj>
    $<TARGET_OBJECTS:examples_common_ptex_obj>
)

target_link_libraries(dxPtexViewer
    ${PLATFORM_LIBRARIES}
)

install(TARGETS dxPtexViewer DESTINATION "${CMAKE_BINDIR_BASE}")
