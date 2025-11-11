//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV_EXAMPLES_D3D11_COMPILE_H
#define OPENSUBDIV_EXAMPLES_D3D11_COMPILE_H

#include <D3DCompiler.h>

namespace D3D11Utils {

ID3DBlob *CompileShader(const char *src, const char *entry, const char *spec);

}

#endif  // OPENSUBDIV_EXAMPLES_D3D11_COMPILE_H
