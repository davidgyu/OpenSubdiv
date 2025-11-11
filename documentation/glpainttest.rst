..  
     Copyright 2013 Pixar
  
     Licensed under the terms set forth in the LICENSE.txt file available at
     https://opensubdiv.org/license.
  

glPaintTest
-----------

.. contents::
   :local:
   :backlinks: none

SYNOPSIS
========

.. parsed-literal:: 
   :class: codefhead

   **glPaintTest** [**-f**] [**-yup**] [**-l** *adaptive refinement level*] 
       *objfile(s)* [**-catmark**] [**-loop**]

   
DESCRIPTION
===========

``glPaintTest`` is a small stand-alone application showing the potential of
using GPU limit tessellation for painting and sculpting applications.

.. image:: images/painttest.jpg
   :width: 400px
   :align: center
   :target: images/painttest.jpg

OPTIONS
=======

See the description of the
`common comand line options <code_examples.html#common-command-line-options>`__
for the subset of common options supported here.

KEYBOARD CONTROLS
=================

   .. code:: c++
   
      c      : use texture as color
      d      : use texture as displacement


.. include:: examples_see_also.rst
