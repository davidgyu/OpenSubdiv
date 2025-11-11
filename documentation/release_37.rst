..
     Copyright 2025 Pixar

     Licensed under the terms set forth in the LICENSE.txt file available at
     https://opensubdiv.org/license.


Overview of Release 3.7
=======================

.. contents::
   :local:
   :backlinks: none

Updates
-------

OpenSubdiv is now licensed under the Tomorrow Open Source Technology license. This is a renaming of the license, but the terms and conditions are the same as before. See [this forum post](https://groups.google.com/g/opensubdiv/c/VggtIKVJf28) for details.

The Osd classes which interface with OpenGL no longer depend on being able to include the platform OpenGL headers. Instead these classes define local type aliases for OpenGL object handles and shader locations.

Other Changes
-------------

Deprecation Announcements
~~~~~~~~~~~~~~~~~~~~~~~~~
    - The header file opensubdiv/osd/opengl.h is no longer used within the code base and may be removed from future releases.

Improvements
~~~~~~~~~~~~
    - Improved Bfr::Paramaterization conversions to sub-face coordinates (GitHub #1366)
    - Improved minimal Bfr::Tessellation patterns for quads (GitHub #1372)

Bug Fixes
~~~~~~~~~
    - Fixed build errors on some platforms by removing use of GL types from headers (GitHub #1364)
    - Fixed build_osd.py to improve compatibility with recent versions of CMake and Python (GitHub #1371, #1373)
