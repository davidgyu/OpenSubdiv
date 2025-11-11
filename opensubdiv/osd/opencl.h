//
//   Copyright 2014 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OPENSUBDIV3_OSD_OPENCL_H
#define OPENSUBDIV3_OSD_OPENCL_H

#if defined(OPENSUBDIV_HAS_CLEW)
#  include <clew.h>
#else
#  if defined(__APPLE__)
#    include <OpenCL/opencl.h>
#  else
#    include <CL/opencl.h>
#  endif
#endif

#endif  // OPENSUBDIV3_OSD_CL_UTIL_H
