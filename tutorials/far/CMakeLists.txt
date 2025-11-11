#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#
macro(osd_add_far_tutorial NAME)

     osd_add_executable(${NAME} "tutorials/far"
        ${ARGN}
        $<TARGET_OBJECTS:sdc_obj>
        $<TARGET_OBJECTS:vtr_obj>
        $<TARGET_OBJECTS:far_obj>
     )

     install(TARGETS ${NAME} DESTINATION "${CMAKE_BINDIR_BASE}/tutorials")

endmacro()


set(TUTORIALS
    tutorial_1_1
    tutorial_1_2
    tutorial_2_1
    tutorial_2_2
    tutorial_2_3
    tutorial_3_1
    tutorial_4_1
    tutorial_4_2
    tutorial_4_3
    tutorial_5_1
    tutorial_5_2
    tutorial_5_3
)

foreach(tutorial ${TUTORIALS})

    add_subdirectory("${tutorial}")

    list(APPEND TUTORIAL_TARGETS "far_${tutorial}")

    add_test(far_${tutorial} ${EXECUTABLE_OUTPUT_PATH}/far_${tutorial})

endforeach()

add_custom_target(far_tutorials DEPENDS ${TUTORIAL_TARGETS})

set_target_properties(far_tutorials
    PROPERTIES 
        FOLDER "tutorials/far"
)
