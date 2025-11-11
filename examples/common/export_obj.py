#
#   Copyright 2013 Pixar
#
     Licensed under the terms set forth in the LICENSE.txt file available at
     https://opensubdiv.org/license.
#

import maya.OpenMaya as OpenMaya

selectionList = OpenMaya.MSelectionList()
OpenMaya.MGlobal.getActiveSelectionList(selectionList)

path = OpenMaya.MDagPath()
selectionList.getDagPath(0, path)

meshFn = OpenMaya.MFnMesh(path)

points = OpenMaya.MPointArray()
normals = OpenMaya.MFloatVectorArray()
u = OpenMaya.MFloatArray()
v = OpenMaya.MFloatArray()
meshFn.getPoints(points)
meshFn.getVertexNormals(True, normals);
meshFn.getUVs(u, v, "map1")
vertexCount = OpenMaya.MIntArray()
vertexList = OpenMaya.MIntArray()
meshFn.getVertices(vertexCount, vertexList)
edgeIds = OpenMaya.MUintArray()
edgeCreaseData = OpenMaya.MDoubleArray()
vtxIds = OpenMaya.MUintArray()
vtxCreaseData = OpenMaya.MDoubleArray()
try:
    meshFn.getCreaseEdges(edgeIds, edgeCreaseData)
except:
    pass
try:
    meshFn.getCreaseVertices(vtxIds, vtxCreaseData)
except:
    pass

f = open('out.obj', 'w')

for i in range(0,points.length()):
    f.write('v %f %f %f\n' % (points[i].x, points[i].y, points[i].z))

for i in range(0,u.length()):
    f.write('vt %f %f \n' % (u[i], v[i]))

for i in range(0,normals.length()):
    f.write('vn %f %f %f\n' % (normals[i].x, normals[i].y, normals[i].z))

f.write('s off\n')

vindex = 0
for i in range(0,vertexCount.length()):
    f.write('f')
    for j in range(0, vertexCount[i]):
        v = vertexList[vindex] + 1

        sutil = OpenMaya.MScriptUtil()
        sutil.createFromInt(0)
        uvptr = sutil.asIntPtr()
        meshFn.getPolygonUVid(i, j, uvptr, 'map1')
        uv = sutil.getInt(uvptr) + 1

        f.write(' %d/%d/%d' % (v, uv, v))
        vindex = vindex+1
    f.write('\n')

if vtxIds.length() > 0:
    f.write('t corner %d/%d/0' % (vtxIds.length(), vtxIds.length()))
    for i in range(0,vtxIds.length()):
        f.write(' %d' % vtxIds[i])
    for i in range(0,vtxCreaseData.length()):
        f.write(' %f' % vtxCreaseData[i])
    f.write('\n')

for i in range(0, edgeIds.length()):
    edgeIt = OpenMaya.MItMeshEdge(path)
    dummy = OpenMaya.MScriptUtil().asIntPtr()
    edgeIt.setIndex(edgeIds[i], dummy)
    faceList = OpenMaya.MIntArray()
    edgeIt.getConnectedFaces(faceList)
    vid0 = edgeIt.index(0)
    vid1 = edgeIt.index(1)

    f.write('t crease 2/1/0 %d %d %f\n' % (vid0, vid1, edgeCreaseData[i]))


f.close()

