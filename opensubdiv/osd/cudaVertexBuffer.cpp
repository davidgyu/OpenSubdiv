//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../osd/cudaVertexBuffer.h"

#include <cuda_runtime.h>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

CudaVertexBuffer::CudaVertexBuffer(int numElements, int numVertices)
    : _numElements(numElements),
      _numVertices(numVertices),
      _cudaMem(0) {
}

CudaVertexBuffer::~CudaVertexBuffer() {
    if (_cudaMem) cudaFree(_cudaMem);
}

CudaVertexBuffer *
CudaVertexBuffer::Create(int numElements, int numVertices,
                         void * /*deviceContext */) {
    CudaVertexBuffer *instance =
        new CudaVertexBuffer(numElements, numVertices);
    if (instance->allocate()) return instance;
    delete instance;
    return NULL;
}

void
CudaVertexBuffer::UpdateData(const float *src, int startVertex, int numVertices,
                             void * /*deviceContext*/) {

    size_t size = _numElements * numVertices * sizeof(float);

    cudaMemcpy((float*)_cudaMem + _numElements * startVertex,
               src, size, cudaMemcpyHostToDevice);
}

int
CudaVertexBuffer::GetNumElements() const {

    return _numElements;
}

int
CudaVertexBuffer::GetNumVertices() const {

    return _numVertices;
}

float *
CudaVertexBuffer::BindCudaBuffer() {

    return static_cast<float*>(_cudaMem);
}

bool
CudaVertexBuffer::allocate() {
    int size = _numElements * _numVertices * sizeof(float);

    cudaError_t err = cudaMalloc(&_cudaMem, size);

    if (err != cudaSuccess) return false;
    return true;
}

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
}  // end namespace OpenSubdiv

