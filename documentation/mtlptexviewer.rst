..  
     Copyright 2019 Pixar
  
     Licensed under the terms set forth in the LICENSE.txt file available at
     https://opensubdiv.org/license.
  

mtlPtexViewer
-------------

.. contents::
   :local:
   :backlinks: none

SYNOPSIS
========

.. parsed-literal::
   :class: codefhead

   **open** **mtlPtexViewer.app** **--args** [**-yup**] [**-u**] [**-a**] [**-l** *isolation level*]
       *ptex color file*
       *ptex displacement file*

DESCRIPTION
===========

``mtlPtexViewer`` is a stand-alone application demonstrating shading with color and displacement ptex maps. Multiple controls are available to experiment with the algorithms.

.. image:: images/mtlptexviewer.png
   :width: 400px
   :align: center
   :target: images/mtlptexviewer.png

KEYBOARD CONTROLS
=================

   .. code:: c++

      q      : quit
      f      : fit frame
      +/-    : increase / decrease tessellation rate

.. include:: examples_see_also.rst
