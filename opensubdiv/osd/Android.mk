#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

# This file is used by the Android NDK to allow OpenSubdiv libraries
# to be imported by client modules.

LOCAL_PATH:= $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE = OpenSubdivOsdCPU
LOCAL_SRC_FILES = libs/armeabi-v7a/libosdCPU.so
LOCAL_EXPORT_C_INCLUDES = $(LOCAL_PATH)/include

include $(PREBUILT_SHARED_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE = OpenSubdivOsdGPU
LOCAL_SRC_FILES = libs/armeabi-v7a/libosdGPU.so
LOCAL_EXPORT_C_INCLUDES = $(LOCAL_PATH)/include

include $(PREBUILT_SHARED_LIBRARY)

