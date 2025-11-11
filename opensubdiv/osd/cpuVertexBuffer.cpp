//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../osd/cpuVertexBuffer.h"

#include <string.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

CpuVertexBuffer::CpuVertexBuffer(int numElements, int numVertices)
    : _numElements(numElements),
      _numVertices(numVertices),
      _cpuBuffer(NULL) {

    _cpuBuffer = new float[numElements * numVertices];
}

CpuVertexBuffer::~CpuVertexBuffer() {

    delete[] _cpuBuffer;
}

CpuVertexBuffer *
CpuVertexBuffer::Create(int numElements, int numVertices,
                        void * /*deviceContext*/) {

    return new CpuVertexBuffer(numElements, numVertices);
}

void
CpuVertexBuffer::UpdateData(const float *src, int startVertex, int numVertices,
                            void * /*deviceContext*/) {

    memcpy(_cpuBuffer + startVertex * _numElements,
           src, GetNumElements() * numVertices * sizeof(float));
}

int
CpuVertexBuffer::GetNumElements() const {

    return _numElements;
}

int
CpuVertexBuffer::GetNumVertices() const {

    return _numVertices;
}

float*
CpuVertexBuffer::BindCpuBuffer() {

    return _cpuBuffer;
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

