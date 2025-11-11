//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_GL_HUD_H
#define OPENSUBDIV_EXAMPLES_GL_HUD_H

#include "glLoader.h"

#include "hud.h"

class GLhud : public Hud {

public:
    GLhud();
    ~GLhud();

    virtual void Init(int width, int height, int framebufferWidth, int framebufferHeight);

    virtual void Rebuild(int width, int height,
                         int framebufferWidth, int framebufferHeight);

    virtual bool Flush();

    GLuint GetFontTexture() const {
        return _fontTexture;
    }

    void FillBackground();

private:
    GLuint _fontTexture;
    GLuint _vbo, _staticVbo;
    GLuint _vao, _staticVao, _bgVao;
    int _staticVboSize;

    GLint _program;
    GLint _mvpMatrix;
    GLint _aPosition, _aColor, _aUV;
    GLint _bgProgram;
};

#endif  // OPENSUBDIV_EXAMPLES_GL_HUD_H
