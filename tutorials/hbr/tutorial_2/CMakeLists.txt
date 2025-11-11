#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

set(SOURCE_FILES
    hbr_tutorial_2.cpp
)

osd_add_executable(hbr_tutorial_2 "tutorials/hbr"
    ${SOURCE_FILES}
)

install(TARGETS hbr_tutorial_2 DESTINATION "${CMAKE_BINDIR_BASE}/tutorials")

