#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

add_subdirectory(common)

if (NOT NO_REGRESSION)

    add_subdirectory(bfr_evaluate)

    add_subdirectory(hbr_regression)

    add_subdirectory(far_regression)

    add_subdirectory(far_perf)

    if(OPENGL_FOUND AND GLFW_FOUND)
        add_subdirectory(osd_regression)
    endif()

endif()
