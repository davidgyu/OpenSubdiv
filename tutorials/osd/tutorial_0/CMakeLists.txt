#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

set(SOURCE_FILES
    osd_tutorial_0.cpp
)

osd_add_executable(osd_tutorial_0 "tutorials/osd"
    ${SOURCE_FILES}
)


target_link_libraries(osd_tutorial_0
    osd_static_cpu
)

install(TARGETS osd_tutorial_0 DESTINATION "${CMAKE_BINDIR_BASE}/tutorials")

