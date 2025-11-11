#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

include_directories("${OPENSUBDIV_INCLUDE_DIR}")

osd_add_possibly_cuda_executable(hbr_regression "regression"
   main.cpp
    $<TARGET_OBJECTS:regression_common_obj>
)

# This is what we "should" be using but we need to escape the backslash and cmake
# regex doesn't allow that apparently
#file(TO_NATIVE_PATH "${PROJECT_SOURCE_DIR}/regression/hbr_regression/baseline/" HBR_BASELINE_DIR)

set(HBR_BASELINE_DIR "${PROJECT_SOURCE_DIR}/regression/hbr_regression/baseline/" )

if (WIN32)
    string(REGEX REPLACE "/" "\\\\\\\\" HBR_BASELINE_DIR "${HBR_BASELINE_DIR}")
endif()

add_definitions(
    -DHBR_BASELINE_DIR="${HBR_BASELINE_DIR}"
)


osd_add_executable(hbr_baseline "regression"
   baseline.cpp
    $<TARGET_OBJECTS:regression_common_obj>
)

install(TARGETS hbr_baseline DESTINATION "${CMAKE_BINDIR_BASE}")
install(TARGETS hbr_regression DESTINATION "${CMAKE_BINDIR_BASE}")
