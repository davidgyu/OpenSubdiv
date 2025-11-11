..  
     Copyright 2019 Pixar
  
     Licensed under the terms set forth in the LICENSE.txt file available at
     https://opensubdiv.org/license.
  

mtlViewer
---------

.. contents::
   :local:
   :backlinks: none

SYNOPSIS
========

.. parsed-literal::
   :class: codefhead

   **open** **mtlViewer.app** **--args** [**-yup**] [**-u**] [**-a**] [**-l** *refinement level*]
      *objfile(s)* [**-catmark**] [**-loop**] [**-bilinear**]

DESCRIPTION
===========

``mtlViewer`` is a stand-alone application that showcases the application of
uniform and feature adaptive subdivision schemes to a collection of geometric
shapes. Multiple controls are available to experiment with the algorithms.

+----------------------------------------------------+
| .. image:: images/mtlviewer.png                    |
|    :width: 400px                                   |
|    :align: center                                  |
|    :target: images/mtlviewer.png                   |
|                                                    |
| mtlViewer running on macOS                         |
+----------------------------------------------------+

+----------------------------------------------------+
| .. image:: images/mtlviewer_ios.png                |
|    :width: 400px                                   |
|    :align: center                                  |
|    :target: images/mtlviewer_ios.png               |
|                                                    |
| mtlViewer running on iOS                           |
+----------------------------------------------------+

KEYBOARD CONTROLS
=================

   .. code:: c++

      q      : quit
      f      : fit frame
      +/-    : increase / decrease tessellation rate

.. include:: examples_see_also.rst
