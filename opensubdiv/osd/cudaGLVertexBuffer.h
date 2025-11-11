//
//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_CUDA_GL_VERTEX_BUFFER_H
#define OPENSUBDIV3_OSD_CUDA_GL_VERTEX_BUFFER_H

#include "../version.h"

#include <cstddef>
#include <cuda_runtime.h>

#include <cuda_gl_interop.h>

namespace OpenSubdiv {
namespace OPENSUBDIV_VERSION {

namespace Osd {

/// \brief Concrete vertex buffer class for cuda subdivision and OpenGL drawing.
///
/// CudaGLVertexBuffer implements CudaVertexBufferInterface and
/// GLVertexBufferInterface.
///
/// The buffer interop between Cuda and GL is handled automatically when a
/// client calls BindCudaBuffer and BindVBO methods.
///
class CudaGLVertexBuffer {
public:
    using ID = unsigned int;    // GLuint resource ID

    /// Creator. Returns NULL if error.
    static CudaGLVertexBuffer * Create(int numElements, int numVertices,
                                       void *deviceContext = NULL);

    /// Destructor.
    ~CudaGLVertexBuffer();

    /// This method is meant to be used in client code in order to provide
    /// coarse vertices data to Osd.
    void UpdateData(const float *src, int startVertex, int numVertices,
                    void *deviceContext = NULL);

    /// Returns how many elements defined in this vertex buffer.
    int GetNumElements() const;

    /// Returns how many vertices allocated in this vertex buffer.
    int GetNumVertices() const;

    /// Returns cuda memory. GL buffer will be mapped to cuda resource
    /// if necessary.
    float * BindCudaBuffer();

    /// Returns the GL buffer object. If the buffer is mapped as a cuda
    /// resource, it will be unmapped back to GL.
    ID BindVBO(void *deviceContext = NULL);

protected:
    /// Constructor.
    CudaGLVertexBuffer(int numElements, int numVertices);

    /// Allocates VBO for this buffer and register as a cuda resource.
    /// Returns true if success.
    bool allocate();

    /// Acquire a cuda resource from GL.
    void map();

    /// Release a cuda resource to GL.
    void unmap();

private:
    int _numElements;
    int _numVertices;
    ID _vbo;
    void *_devicePtr;
    struct cudaGraphicsResource *_cudaResource;
};

}  // end namespace Osd

}  // end namespace OPENSUBDIV_VERSION
using namespace OPENSUBDIV_VERSION;

}  // end namespace OpenSubdiv

#endif  // OPENSUBDIV3_OSD_CUDA_GL_VERTEX_BUFFER_H
