//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../osd/mtlVertexBuffer.h"
#include <Metal/Metal.h>
#include <TargetConditionals.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

CPUMTLVertexBuffer::CPUMTLVertexBuffer(int numElements, int numVertices)
:
_numElements(numElements), _numVertices(numVertices),
_buffer(nullptr), _dirty(true)
{

}

bool CPUMTLVertexBuffer::allocate(MTLContext* context)
{
#if TARGET_OS_IOS || TARGET_OS_TV
    _buffer = [context->device newBufferWithLength: _numElements * _numVertices * sizeof(float) options:MTLResourceOptionCPUCacheModeDefault];
#elif TARGET_OS_OSX
    _buffer = [context->device newBufferWithLength: _numElements * _numVertices * sizeof(float) options:MTLResourceStorageModeManaged];
#endif
    if(_buffer == nil)
        return false;

    _dirty = true;
    _buffer.label = @"OSD VertexBuffer";

    return true;
}

CPUMTLVertexBuffer* CPUMTLVertexBuffer::Create(int numElements, int numVertices, MTLContext* context)
{
    auto instance = new CPUMTLVertexBuffer(numElements, numVertices);
    if(!instance->allocate(context))
    {
        delete instance;
        return nullptr;
    }

    return instance;
}

void CPUMTLVertexBuffer::UpdateData(const float* src, int startVertex, int numVertices, MTLContext*)
{
    _dirty = true;
    memcpy(((float*)_buffer.contents) + startVertex * _numElements, src, _numElements * numVertices * sizeof(float));
}

float* CPUMTLVertexBuffer::BindCpuBuffer()
{
    _dirty = true;
    return (float*)_buffer.contents;
}

id<MTLBuffer> CPUMTLVertexBuffer::BindMTLBuffer(MTLContext*)
{
#if TARGET_OS_OSX
    if(_dirty)
        [_buffer didModifyRange:NSMakeRange(0, _buffer.length)];
    _dirty = false;
#endif
    return _buffer;
}



} //end namepsace Osd

} //end namespace OPENSUBDIV_VERSION
} //end namespace OpenSubdiv
