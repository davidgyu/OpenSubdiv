//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_GL_PTEX_MIPMAP_TEXTURE_H
#define OPENSUBDIV_EXAMPLES_GL_PTEX_MIPMAP_TEXTURE_H

#include "glLoader.h"

#include <opensubdiv/osd/nonCopyable.h>

#include <Ptexture.h>
#include <stdlib.h>

class GLPtexMipmapTexture : OpenSubdiv::Osd::NonCopyable<GLPtexMipmapTexture> {
public:
    static GLPtexMipmapTexture * Create(PtexTexture * reader,
                                           int maxLevels=-1,
                                           size_t targetMemory=0);

    /// Returns GLSL shader snippet to fetch ptex
    static const char *GetShaderSource();

    /// Returns the texture buffer containing the layout of the ptex faces
    /// in the texels texture array.
    GLuint GetLayoutTextureBuffer() const { return _layout; }

    /// Returns the texels texture array.
    GLuint GetTexelsTexture() const { return _texels; }

    /// Returns the amount of allocated memory (in bytes)
    size_t GetMemoryUsage() const { return _memoryUsage; }

    ~GLPtexMipmapTexture();

private:
    GLPtexMipmapTexture();

    GLsizei _width,   // width / height / depth of the 3D texel buffer
            _height,
            _depth;

    GLint   _format;  // texel color format

    GLuint _layout,   // per-face lookup table
           _texels;   // texel data

    size_t _memoryUsage;  // total amount of memory used (estimate)
};

#endif  // OPENSUBDIV_EXAMPLES_GL_PTEX_MIPMAP_TEXTURE_H
