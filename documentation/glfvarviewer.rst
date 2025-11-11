..  
     Copyright 2013 Pixar
  
     Licensed under the terms set forth in the LICENSE.txt file available at
     https://opensubdiv.org/license.
  

glFVarViewer
------------

.. contents::
   :local:
   :backlinks: none

SYNOPSIS
========

.. parsed-literal:: 
   :class: codefhead

   **glFVarViewer** [**-f**] [**-u**] [**-a**] [**-l** *refinement level*] [**-c** *animation loops*]
       *objfile(s)* [**-catmark**] [**-loop**] [**-bilinear**]


DESCRIPTION
===========

``glFVarViewer`` is a stand-alone application that allows the inspection of
face-varying data interpolation. The window displays 2 views:

    * left side: regular 3D view of the model, with a procedural (u,v) texture
    * right side: a 2D view of the face-varying (u,v) pair

The HUD allows interactive manipulation of the various face-varying boundary
interpolation rules and tags.

.. image:: images/uvviewer.jpg 
   :width: 400px
   :align: center
   :target: images/uvviewer.jpg 

OPTIONS
=======

See the description of the
`common comand line options <code_examples.html#common-command-line-options>`__
for the subset of common options supported here.

.. include:: examples_see_also.rst
