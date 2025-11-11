#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
    "${OPENGL_LOADER_INCLUDE_DIRS}"
    "${GLFW_INCLUDE_DIR}"
)

set(SOURCE_FILES
    main.cpp
)

set(PLATFORM_LIBRARIES
    "${OSD_LINK_TARGET}"
    "${OPENGL_LOADER_LIBRARIES}"
    "${GLFW_LIBRARIES}"
)

osd_add_executable(osd_regression "regression"
    "${SOURCE_FILES}"
    $<TARGET_OBJECTS:regression_common_obj>
)

target_link_libraries(osd_regression
    ${PLATFORM_LIBRARIES}
)

install(TARGETS osd_regression DESTINATION "${CMAKE_BINDIR_BASE}")

if (NOT NO_GLTESTS)
    add_test(osd_regression ${EXECUTABLE_OUTPUT_PATH}/osd_regression)
endif()
