#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#


include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}/"
)

add_subdirectory(hbr)

add_subdirectory(far)

add_subdirectory(bfr)

add_subdirectory(osd)

add_custom_target(tutorials
    DEPENDS
        hbr_tutorials
        far_tutorials
        bfr_tutorials
        osd_tutorials
)
