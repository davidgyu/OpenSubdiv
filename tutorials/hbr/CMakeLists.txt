#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

set(TUTORIALS
    tutorial_0
    tutorial_1
    tutorial_2
)

foreach(tutorial ${TUTORIALS})

    add_subdirectory("${tutorial}")

    list(APPEND TUTORIAL_TARGETS "hbr_${tutorial}")

endforeach()

add_custom_target(hbr_tutorials DEPENDS ${TUTORIAL_TARGETS})

set_target_properties(hbr_tutorials
    PROPERTIES 
        FOLDER "tutorials/hbr"
)
