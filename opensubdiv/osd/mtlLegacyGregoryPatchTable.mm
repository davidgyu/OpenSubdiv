//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../osd/mtlLegacyGregoryPatchTable.h"
#include <Metal/Metal.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {
namespace Osd {

static id<MTLBuffer> createBuffer(const void* data, const size_t length,
                                  MTLContext* context)
{
  @autoreleasepool {
    auto cmdBuf = [context->commandQueue commandBuffer];
    auto blitEncoder = [cmdBuf blitCommandEncoder];

    auto stageBuffer = [context->device newBufferWithBytes:data length:length options:MTLResourceCPUCacheModeDefaultCache];

    auto finalBuffer = [context->device newBufferWithLength:length options:MTLResourceStorageModePrivate];

    [blitEncoder copyFromBuffer:stageBuffer sourceOffset:0 toBuffer:finalBuffer destinationOffset:0 size:length];
    [blitEncoder endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

#if !__has_feature(objc_arc)
      [stageBuffer release];
#endif

    return finalBuffer;
  }
}

MTLLegacyGregoryPatchTable::~MTLLegacyGregoryPatchTable()
{

}

MTLLegacyGregoryPatchTable* MTLLegacyGregoryPatchTable::Create(const Far::PatchTable *farPatchTable,
                                                               MTLContext* context)
{
    auto pt = new MTLLegacyGregoryPatchTable();
    auto& vertexValenceTable = farPatchTable->GetVertexValenceTable();
    auto& quadOffsetsTable = farPatchTable->GetQuadOffsetsTable();

    if(!vertexValenceTable.empty())
    {
        pt->_vertexValenceBuffer = createBuffer(vertexValenceTable.data(), vertexValenceTable.size() * sizeof(vertexValenceTable[0]), context);
    }

    if(!quadOffsetsTable.empty())
    {
        pt->_quadOffsetsBuffer = createBuffer(quadOffsetsTable.data(), quadOffsetsTable.size() * sizeof(quadOffsetsTable[0]), context);
    }

    pt->_quadOffsetsBase[0] = 0;
    pt->_quadOffsetsBase[1] = 0;

    for(auto i = 0; i < farPatchTable->GetNumPatchArrays(); i++)
    {
        if(farPatchTable->GetPatchArrayDescriptor(i).GetType() == Far::PatchDescriptor::GREGORY)
        {
            pt->_quadOffsetsBase[1] = farPatchTable->GetNumPatches(i) * 4;
            break;
        }
    }

    return pt;
}

void MTLLegacyGregoryPatchTable::UpdateVertexBuffer(id<MTLBuffer> vbo, int, int, MTLContext*)
{
    _vertexBuffer = vbo;
}

} //end namespace Osd

} //end namespace OPENSUBDIV_VERSION
} //end namespace OpenSubdiv
