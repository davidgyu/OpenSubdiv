//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_GL_UTILS_H
#define OPENSUBDIV_EXAMPLES_GL_UTILS_H

#include "glLoader.h"

#include <cstdio>
#include <string>
#include <iostream>

namespace GLUtils {

void InitializeGL();

void SetMinimumGLVersion(int argc=0, char **argv=NULL);

void PrintGLVersion();

void CheckGLErrors(std::string const & where = "");

GLuint CompileShader(GLenum shaderType, const char *source);

void WriteScreenshot(int width, int height);

bool SupportsAdaptiveTessellation();

// Helper function that parses the opengl version string, retrieving the
// major and minor version from it.
void GetMajorMinorVersion(int *major, int *minor);

// Gets the shader version based on the current opengl version and returns 
// it in a string form.
std::string GetShaderVersion();

std::string GetShaderVersionInclude();

bool GL_ARBSeparateShaderObjectsOrGL_VERSION_4_1();

bool GL_ARBComputeShaderOrGL_VERSION_4_3();

};

#endif  // OPENSUBDIV_EXAMPLES_GL_UTILS_H


