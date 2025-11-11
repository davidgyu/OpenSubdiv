#
#   Copyright 2013 Pixar
#
#   Licensed under the terms set forth in the LICENSE.txt file available at
#   https://opensubdiv.org/license.
#

#-------------------------------------------------------------------------------
set(PUBLIC_HEADER_FILES
    allocator.h
    bilinear.h
    catmark.h
    cornerEdit.h
    creaseEdit.h
    faceEdit.h
    face.h
    fvarData.h
    fvarEdit.h
    halfedge.h
    hierarchicalEdit.h
    holeEdit.h
    loop.h
    mesh.h
    subdivision.h
    vertexEdit.h
    vertex.h
)

#-------------------------------------------------------------------------------

install( 
    FILES
        ${PUBLIC_HEADER_FILES}
    DESTINATION 
        "${CMAKE_INCDIR_BASE}/hbr"
    PERMISSIONS
        OWNER_READ
        GROUP_READ
        WORLD_READ )

#-------------------------------------------------------------------------------
