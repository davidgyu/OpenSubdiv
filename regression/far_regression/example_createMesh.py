#!/usr/bin/env python
#
#   Copyright 2013 Pixar
#
     Licensed under the terms set forth in the LICENSE.txt file available at
     https://opensubdiv.org/license.
#

"""
Example

Description:
    Create a Maya Mesh using OpenMaya

    in Maya's python editor :

    import example_createMesh
    reload(example_createMesh)

    example_createMesh.readPolyFile('/host/devel/rtc3/trees/dev/amber/bin/pbr2/script')

"""

from maya import OpenMaya as om
from itertools import chain
import maya.cmds as cmds

def getDagPath(nodeName):
    """Get an MDagPath for the associated node name
    """
    selList = om.MSelectionList()
    selList.add(nodeName)
    dagPath = om.MDagPath()
    selList.getDagPath(0, dagPath)
    return dagPath

def convertToMIntArray(listOfInts):
    newMIntArray = om.MIntArray(len(listOfInts))
    for i in range(len(listOfInts)):
        val = listOfInts[i]
        newMIntArray.set(val, i)
    return newMIntArray


def convertToMPointArray(listOfVertexTuples):
    newMPointArray = om.MPointArray(len(listOfVertexTuples))
    for i in range(len(listOfVertexTuples)):
        v = listOfVertexTuples[i]
        newMPointArray.set(i, v[0], v[1], v[2], 1.0)
    return newMPointArray


def createMesh(vertices, polygons, parent=None):
    '''Create a mesh with the specified vertices and polygons
    '''
    # The parameters used in MFnMesh.create() can all be derived from the
    # input vertices and polygon lists

    numVertices = len(vertices)
    numPolygons = len(polygons)

    verticesM = convertToMPointArray(vertices)

    polygonCounts = [len(i) for i in polygons]
    polygonCountsM = convertToMIntArray(polygonCounts)

    # Flatten the list of lists
    # Reference: http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    polygonConnects = list(chain.from_iterable(polygons))
    polygonConnectsM = convertToMIntArray(polygonConnects)

    # Determine parent
    if parent == None:
        parentM = om.MObject()
    else:
        parentM = getDagPath(parent).node()

    # Create Mesh
    newMesh = om.MFnMesh()
    newTransformOrShape = newMesh.create(
        numVertices,
        numPolygons,
        verticesM,
        polygonCountsM,
        polygonConnectsM,
        parentM)

    dagpath = om.MDagPath()
    om.MDagPath.getAPathTo( newTransformOrShape, dagpath )

    # Assign the default shader to the mesh.
    cmds.sets(
        dagpath.fullPathName(),
        edit=True,
        forceElement='initialShadingGroup')

    return dagpath.partialPathName()

def readPolyFile(path):
    polys = ''
    try:
        with open(path, 'r') as f:
            polys = ''
            for line in f.readlines():
                polys += line.rstrip()
    except:
        print 'Cannot read '+str(path)

    polys = eval(polys)

    tx = 0.0
    ty = 0.0
    for poly in polys:
        verts = poly['verts']
        faces = poly['faces']
        parent = None
        dagpath = createMesh(verts, faces, parent)
        cmds.move( tx, ty, 0, dagpath, absolute=True )
        tx+=4.0
        if tx > 16.0:
            tx=0.0
            ty+=4.0
