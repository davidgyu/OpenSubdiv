//
//   Copyright 2018 DreamWorks Animation LLC.
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

static const std::string bilinear_nonplanar = 
"\n"
"v -0.6 -0.5 -0.4\n"
"v -0.1  0.0  0.4\n"
"v -0.6  0.5 -0.4\n"
"v -1.1  0.0  0.4\n"
"v  0.6 -0.5 -0.4\n"
"v  1.1  0.0  0.4\n"
"v  0.6  0.5 -0.4\n"
"v  0.1  0.0  0.4\n"
"\n"
"vt 0 0\n"
"vt 1 0\n"
"vt 1 1\n"
"vt 0 1\n"
"\n"
"f 1/1 2/2 3/3 4/4\n"
"f 6/2 7/3 8/4 5/1\n"
"\n"
"t interpolateboundary 1/0/0 1\n"
;
