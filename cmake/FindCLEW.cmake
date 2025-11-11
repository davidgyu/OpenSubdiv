#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

# Try to find CLEW library and include path.
# Once done this will define
#
# CLEW_FOUND
# CLEW_INCLUDE_DIR
# CLEW_LIBRARY
#

include (FindPackageHandleStandardArgs)

if(WIN32)
  set(_clew_SEARCH_DIRS
    "${CLEW_LOCATION}/include"
    "$ENV{CLEW_LOCATION}/include"
    "$ENV{PROGRAMFILES}/CLEW/include"
    "${PROJECT_SOURCE_DIR}/extern/clew/include"
  )
else()
  set(_clew_SEARCH_DIRS
      "${CLEW_LOCATION}"
      "$ENV{CLEW_LOCATION}"
      /usr
      /usr/local
      /sw
      /opt/local
      /opt/lib/clew
  )
endif()

find_path(CLEW_INCLUDE_DIR
  NAMES
    clew.h
  HINTS
    ${_clew_SEARCH_DIRS}
  PATH_SUFFIXES
    include
  NO_DEFAULT_PATH
  DOC "The directory where clew.h resides")

find_library(CLEW_LIBRARY
  NAMES
    CLEW clew
  PATHS
    ${_clew_SEARCH_DIRS}
  PATH_SUFFIXES
    lib lib64
  NO_DEFAULT_PATH
  DOC "The CLEW library")

find_package_handle_standard_args(CLEW
    REQUIRED_VARS
        CLEW_INCLUDE_DIR
        CLEW_LIBRARY
)
