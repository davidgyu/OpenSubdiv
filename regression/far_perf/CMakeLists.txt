#
#   Copyright 2015 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
)

set(SOURCE_FILES
    far_perf.cpp
)

set(PLATFORM_LIBRARIES
    "${OSD_LINK_TARGET}"
)

osd_add_executable(far_perf "regression"
    ${SOURCE_FILES}
    $<TARGET_OBJECTS:sdc_obj>
    $<TARGET_OBJECTS:vtr_obj>
    $<TARGET_OBJECTS:far_obj>
    $<TARGET_OBJECTS:regression_common_obj>
)

install(TARGETS far_perf DESTINATION "${CMAKE_BINDIR_BASE}")

add_test(far_perf ${EXECUTABLE_OUTPUT_PATH}/far_regression)

