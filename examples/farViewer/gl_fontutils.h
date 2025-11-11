//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef GL_FONT_UTILS_H
#define GL_FONT_UTILS_H

#include "../common/glUtils.h"

#include <vector>

class GLFont {
public:

    GLFont(GLuint fontTexture);

    ~GLFont();

    void Draw(GLuint transforUB);

    void Clear();

    void Print3D(float const pos[3], const char * str, int color=0);
    
    void SetFontScale(float scale);

    struct Char {
        float pos[3];
        float ofs[2];
        float alpha;
        float color;
    };
    
    std::vector<Char> & GetChars() {
        _dirty=true;
        return _chars;
    }
    
    
private:

    void bindProgram();

    std::vector<Char> _chars;
    bool _dirty;

    GLuint _program,
           _transformBinding,
           _attrPosition,
           _attrData,
           _fontTexture,
           _scale,
           _VAO,
           _EAO,
           _VBO;
};

#endif // GL_FONT_UTILS_H
