#
#   Copyright 2021 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#
macro(osd_add_bfr_tutorial NAME)

     osd_add_executable(${NAME} "tutorials/bfr"
        ${ARGN}
        $<TARGET_OBJECTS:sdc_obj>
        $<TARGET_OBJECTS:vtr_obj>
        $<TARGET_OBJECTS:far_obj>
        $<TARGET_OBJECTS:bfr_obj>
        $<TARGET_OBJECTS:regression_common_obj>
     )

     install(TARGETS ${NAME} DESTINATION "${CMAKE_BINDIR_BASE}/tutorials")

endmacro()


set(TUTORIALS
    tutorial_1_1
    tutorial_1_2
    tutorial_1_3
    tutorial_1_4
    tutorial_1_5
    tutorial_2_1
    tutorial_2_2
    tutorial_3_1
    tutorial_3_2
)

foreach(tutorial ${TUTORIALS})

    add_subdirectory("${tutorial}")

    list(APPEND TUTORIAL_TARGETS "bfr_${tutorial}")

    add_test(bfr_${tutorial} ${EXECUTABLE_OUTPUT_PATH}/bfr_${tutorial})

endforeach()

add_custom_target(bfr_tutorials DEPENDS ${TUTORIAL_TARGETS})

set_target_properties(bfr_tutorials
    PROPERTIES 
        FOLDER "tutorials/bfr"
)
