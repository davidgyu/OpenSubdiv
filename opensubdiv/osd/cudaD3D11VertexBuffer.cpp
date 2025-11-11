//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../osd/cudaD3D11VertexBuffer.h"
#include "../far/error.h"

#include <D3D11.h>
#include <cuda_runtime.h>
#include <cuda_d3d11_interop.h>
#include <cassert>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

CudaD3D11VertexBuffer::CudaD3D11VertexBuffer(int numElements, int numVertices)
    : _numElements(numElements), _numVertices(numVertices),
      _d3d11Buffer(NULL), _cudaBuffer(NULL), _cudaResource(NULL) {
}

CudaD3D11VertexBuffer::~CudaD3D11VertexBuffer() {

    unmap();
    cudaGraphicsUnregisterResource(_cudaResource);
    _d3d11Buffer->Release();
}

CudaD3D11VertexBuffer *
CudaD3D11VertexBuffer::Create(int numElements, int numVertices,
                              ID3D11DeviceContext *deviceContext) {
    CudaD3D11VertexBuffer *instance =
        new CudaD3D11VertexBuffer(numElements, numVertices);

    ID3D11Device *device;
    deviceContext->GetDevice(&device);
    if (instance->allocate(device)) return instance;
    delete instance;
    return NULL;
}

void
CudaD3D11VertexBuffer::UpdateData(const float *src,
                                  int startVertex, int numVertices,
                                  void * /*deviceContext*/) {

    map();
    cudaMemcpy((float*)_cudaBuffer + _numElements * startVertex,
               src, _numElements * numVertices * sizeof(float),
               cudaMemcpyHostToDevice);
}

int
CudaD3D11VertexBuffer::GetNumElements() const {

    return _numElements;
}

int
CudaD3D11VertexBuffer::GetNumVertices() const {

    return _numVertices;
}

float *
CudaD3D11VertexBuffer::BindCudaBuffer() {

    map();
    return (float*)_cudaBuffer;
}

ID3D11Buffer *
CudaD3D11VertexBuffer::BindD3D11Buffer(ID3D11DeviceContext *deviceContext) {

    unmap();
    return _d3d11Buffer;
}

bool
CudaD3D11VertexBuffer::allocate(ID3D11Device *device) {

    D3D11_BUFFER_DESC hBufferDesc;
    hBufferDesc.ByteWidth           = _numElements * _numVertices * sizeof(float);
    hBufferDesc.Usage               = D3D11_USAGE_DYNAMIC;
    hBufferDesc.BindFlags           = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_SHADER_RESOURCE;
    hBufferDesc.CPUAccessFlags      = D3D11_CPU_ACCESS_WRITE;
    hBufferDesc.MiscFlags           = 0;
    hBufferDesc.StructureByteStride = sizeof(float);

    HRESULT hr;
    hr = device->CreateBuffer(&hBufferDesc, NULL, &_d3d11Buffer);
    if(FAILED(hr)) {
        Far::Error(Far::FAR_RUNTIME_ERROR,
                 "Fail in CreateBuffer\n");
        return false;
    }
    
    // register d3d11buffer as cuda resource
    cudaError_t err = cudaGraphicsD3D11RegisterResource(
        &_cudaResource, _d3d11Buffer, cudaGraphicsRegisterFlagsNone);

    if (err != cudaSuccess) return false;
    return true;
}

void
CudaD3D11VertexBuffer::map() {

    if (_cudaBuffer) return;
    size_t num_bytes;
    void *ptr;
    
    cudaGraphicsMapResources(1, &_cudaResource, 0);
    cudaGraphicsResourceGetMappedPointer(&ptr, &num_bytes, _cudaResource);
    _cudaBuffer = ptr;
}

void
CudaD3D11VertexBuffer::unmap() {
    
    if (_cudaBuffer == NULL) return;
    cudaGraphicsUnmapResources(1, &_cudaResource, 0);
    _cudaBuffer = NULL;
}

}  // end namespace Osd

} // end namespace OPENSUBDIV_VERSION
} // end namespace OpenSubdiv

