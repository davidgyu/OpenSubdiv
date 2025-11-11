#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

if(CMAKE_CROSSCOMPILING)
	set(STRINGIFY_LOCATION "STRINGIFY-NOTFOUND" CACHE FILEPATH "Point it to the stringify binary from a native build") 
	add_executable(stringify IMPORTED GLOBAL)
	set_property(TARGET stringify PROPERTY IMPORTED_LOCATION ${STRINGIFY_LOCATION})
endif()

if(NOT CMAKE_CROSSCOMPILING)
	osd_add_executable(stringify "opensubdiv/tools"
	    main.cpp
	)
	install(TARGETS stringify DESTINATION ${CMAKE_BINDIR_BASE})
endif() 
