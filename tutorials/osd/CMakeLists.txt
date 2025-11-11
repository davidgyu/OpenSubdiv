#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

set(TUTORIALS
    tutorial_0
)

foreach(tutorial ${TUTORIALS})

    add_subdirectory("${tutorial}")

    list(APPEND TUTORIAL_TARGETS "osd_${tutorial}")

endforeach()

add_custom_target(osd_tutorials DEPENDS ${TUTORIAL_TARGETS})

set_target_properties(osd_tutorials
    PROPERTIES 
        FOLDER "tutorials/osd"
)
