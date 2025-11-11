#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

# *** mtlViewer ***

if(IOS)
    set(BUNDLE_FILES
        "iOS/Base.lproj/LaunchScreen.storyboard"
        "iOS/Base.lproj/Main.storyboard"
    )
    set(PLATFORM_PLIST "iOS/Info.plist")
    set(PLATFORM_FILES
        "iOS/AppDelegate.h"
        "iOS/AppDelegate.m"
        "iOS/main.m"
        "iOS/ViewController.h"
        "iOS/ViewController.mm"
    )
    list(APPEND PLATFORM_LIBRARIES
        "-framework UIKit"
    )
    set_source_files_properties("${BUNDLE_FILES}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
else()
    set(BUNDLE_FILES
        "OSX/MainMenu.xib"
    )
    set(PLATFORM_PLIST "OSX/Info.plist")
    set(PLATFORM_FILES
    	"OSX/AppDelegate.h"
    	"OSX/AppDelegate.m"
    	"OSX/main.m"
    	"OSX/ViewController.h"
    	"OSX/ViewController.mm"
    )
    list(APPEND PLATFORM_LIBRARIES
        "-framework OpenGL"
        "-framework AppKit"
    )
    set_source_files_properties("${BUNDLE_FILES}" PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
endif()

include_directories(
    "${OPENSUBDIV_INCLUDE_DIR}"
    "${METAL_INCLUDE_DIR}"
)

list(APPEND PLATFORM_FILES
    "mtlViewer.h"
    "mtlViewer.mm"
)

list(APPEND PLATFORM_LIBRARIES
    "osd_static_framework"
    "${METAL_LIBRARY}"
    "-framework MetalKit"
    "-framework Foundation"
    "-framework QuartzCore"
)

osd_stringify("mtlViewer.metal" INC_FILES)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

osd_add_executable(mtlViewer "examples/mtlViewer"
    MACOSX_BUNDLE
    "${PLATFORM_FILES}"
    "${INC_FILES}"
    "${BUNDLE_FILES}"
    $<TARGET_OBJECTS:regression_common_obj>
    $<TARGET_OBJECTS:regression_far_utils_obj>
    $<TARGET_OBJECTS:examples_common_mtl_obj>
)


set_target_properties (mtlViewer PROPERTIES 
    LINKER_LANGUAGE CXX
    RESOURCE "${BUNDLE_FILES}"
    MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_LIST_DIR}/${PLATFORM_PLIST}"
)

if(IOS)
    set_target_properties(mtlViewer PROPERTIES
        XCODE_ATTRIBUTE_TARGETED_DEVICE_FAMILY "1,2"
    )
endif()

target_link_libraries(mtlViewer
    ${PLATFORM_LIBRARIES}
)

install(TARGETS mtlViewer DESTINATION "${CMAKE_BINDIR_BASE}")
