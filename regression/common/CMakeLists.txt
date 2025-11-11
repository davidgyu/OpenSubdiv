#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

set(REGRESSION_COMMON_SOURCE_FILES
    arg_utils.cpp
    shape_utils.cpp
)

set(REGRESSION_COMMON_HEADER_FILES
    arg_utils.h
    cmp_utils.h
    hbr_utils.h
    shape_utils.h
    far_utils.h
)

include_directories("${OPENSUBDIV_INCLUDE_DIR}")

add_library(regression_common_obj
    OBJECT
        ${REGRESSION_COMMON_SOURCE_FILES}
        ${REGRESSION_COMMON_HEADER_FILES}
)

set_target_properties(regression_common_obj
    PROPERTIES
        FOLDER "regression"
)

add_library(regression_far_utils_obj
    OBJECT
        far_utils.cpp
        far_utils.h
)

set_target_properties(regression_far_utils_obj
    PROPERTIES
        FOLDER "regression"
)

