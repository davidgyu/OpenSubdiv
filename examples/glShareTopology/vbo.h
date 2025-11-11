//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_VBO_H
#define OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_VBO_H

#include "glLoader.h"

#include <opensubdiv/osd/bufferDescriptor.h>

#include <vector>

template <class VERTEX_BUFFER, class DEVICE_CONTEXT>
class VBO {
public:
    VBO(OpenSubdiv::Osd::BufferDescriptor const &vertexDesc,
        OpenSubdiv::Osd::BufferDescriptor const &varyingDesc,
        bool interleaved, int numVertices, DEVICE_CONTEXT *deviceContext) :
        _vertexDesc(vertexDesc),
        _varyingDesc(varyingDesc),
        _numVertices(numVertices),
        _vertexBuffer(NULL), _varyingBuffer(NULL), _interleaved(interleaved),
        _deviceContext(deviceContext) {

        if (interleaved) {
            assert(vertexDesc.stride == varyingDesc.stride);
            _vertexBuffer = createVertexBuffer(
                vertexDesc.stride, numVertices);
        } else {
            if (vertexDesc.stride > 0) {
                _vertexBuffer = createVertexBuffer(
                    vertexDesc.stride, numVertices);
            }
            if (varyingDesc.stride > 0) {
                _varyingBuffer = createVertexBuffer(
                    varyingDesc.stride, numVertices);
            }
        }
    }

    ~VBO() {
        delete _vertexBuffer;
        delete _varyingBuffer;
    }

    OpenSubdiv::Osd::BufferDescriptor const &GetVertexDesc() const {
        return _vertexDesc;
    }
    OpenSubdiv::Osd::BufferDescriptor const &GetVaryingDesc() const {
        return _varyingDesc;
    }

    void UpdateVertexBuffer(int vertsOffset, std::vector<float> const &src) {
        updateVertexBuffer(_vertexBuffer, &src[0], vertsOffset,
                           (int)src.size()/_vertexBuffer->GetNumElements());
    }
    void UpdateVaryingBuffer(int vertsOffset, std::vector<float> const &src) {
        updateVertexBuffer(_varyingBuffer, &src[0], vertsOffset,
                           (int)src.size()/_varyingBuffer->GetNumElements());
    }

    GLuint BindVertexBuffer() {
        return _vertexBuffer->BindVBO();
    }

    GLuint BindVaryingBuffer() {
        return _varyingBuffer->BindVBO();
    }

    VERTEX_BUFFER *GetVertexBuffer() const {
        return _vertexBuffer;
    }

    VERTEX_BUFFER *GetVaryingBuffer() const {
        return _interleaved ? _vertexBuffer : _varyingBuffer;
    }

    size_t GetSize() const {
        size_t size = _numVertices * _vertexDesc.stride;
        if (_varyingBuffer) size += _numVertices * _varyingDesc.stride;
        return size * sizeof(float);
    }

private:
    VERTEX_BUFFER *createVertexBuffer(int numElements, int numVertices) {
        return VERTEX_BUFFER::Create(numElements, numVertices, _deviceContext);
    }

    void updateVertexBuffer(VERTEX_BUFFER *vertexBuffer,
                            const float *src, int startVertex,
                            int numVertices) {
        vertexBuffer->UpdateData(src, startVertex, numVertices, _deviceContext);
    }

    OpenSubdiv::Osd::BufferDescriptor _vertexDesc;
    OpenSubdiv::Osd::BufferDescriptor _varyingDesc;

    // # of vertices total, including both control verts and refined verts.
    int _numVertices;

    VERTEX_BUFFER *_vertexBuffer;
    VERTEX_BUFFER *_varyingBuffer;
    bool _interleaved;
    DEVICE_CONTEXT *_deviceContext;
};

#endif  //  OPENSUBDIV_EXAMPLES_GL_SHARE_TOPOLOGY_VBO_H
