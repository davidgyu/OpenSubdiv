#
#   Copyright 2021 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

include_directories("${OPENSUBDIV_INCLUDE_DIR}")

set(SOURCE_FILES
    main.cpp
    bfrSurfaceEvaluator.cpp
    farPatchEvaluator.cpp
)

set(PLATFORM_LIBRARIES
    "${OSD_LINK_TARGET}"
)

osd_add_executable(bfr_evaluate "regression"
    ${SOURCE_FILES}
    $<TARGET_OBJECTS:sdc_obj>
    $<TARGET_OBJECTS:vtr_obj>
    $<TARGET_OBJECTS:far_obj>
    $<TARGET_OBJECTS:bfr_obj>
    $<TARGET_OBJECTS:regression_common_obj>
)

install(TARGETS bfr_evaluate DESTINATION "${CMAKE_BINDIR_BASE}")

add_test(bfr_evaluate_pos ${EXECUTABLE_OUTPUT_PATH}/bfr_evaluate
                          -all -silent -l 3 -pass 2 -d1)
add_test(bfr_evaluate_uv1 ${EXECUTABLE_OUTPUT_PATH}/bfr_evaluate
                          -all -silent -l 3 -pass 0 -skippos -uv -uvint 1)
add_test(bfr_evaluate_uv5 ${EXECUTABLE_OUTPUT_PATH}/bfr_evaluate
                          -all -silent -l 3 -pass 0 -skippos -uv -uvint 5)

