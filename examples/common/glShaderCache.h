//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_GL_SHADER_CACHE_H
#define OPENSUBDIV_EXAMPLES_GL_SHADER_CACHE_H

#include "glLoader.h"

#include <map>
#include <string>
#include "./shaderCache.h"

class GLDrawConfig {
public:
    explicit GLDrawConfig(const std::string &version);
    ~GLDrawConfig();

    bool CompileAndAttachShader(GLenum shaderType, const std::string &source);
    bool Link();

    GLuint GetProgram() const {
        return _program;
    }

private:
    GLuint _program;
    std::string _version;
    int _numShaders;
};

// workaround for template alias
#if 0
template <typename DESC_TYPE>
using GLShaderCache = ShaderCacheT<DESC_TYPE, GLDrawConfig>;
#else
template <typename DESC_TYPE>
class GLShaderCache : public ShaderCacheT<DESC_TYPE, GLDrawConfig> {
};
#endif


#endif  // OPENSUBDIV_EXAMPLES_GL_SHADER_CACHE_H
