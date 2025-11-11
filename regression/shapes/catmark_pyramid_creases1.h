//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

static const std::string catmark_pyramid_creases1 =
"# This file uses centimeters as units for non-parametric coordinates.\n"
"\n"
"v 0.0 0.0 2.0\n"
"v 0.0 -2.0 0.0\n"
"v 2.0 0.0 0.0\n"
"v 0.0 2.0 0.0\n"
"v -2.0 0.0 0.0\n"
"vt 0.0 0.0\n"
"vt 0.0 -2.0\n"
"vt 2.0 0.0\n"
"vt 0.0 2.0\n"
"vt -2.0 0.0\n"
"s off\n"
"f 1/1 2/2 3/3\n"
"f 1/1 3/3 4/4\n"
"f 1/1 4/4 5/5\n"
"f 1/1 5/5 2/2\n"
"f 5/5 4/4 3/3 2/2\n"
"t crease 2/1/0 1 2 2.5\n"
"t crease 2/1/0 2 3 2.5\n"
"t crease 2/1/0 3 4 2.5\n"
"t crease 2/1/0 4 1 2.5\n"
"t corner 1/1/0 0 3.5\n"
"t corner 1/1/0 1 2.5\n"
"t interpolateboundary 1/0/0 2\n"
"\n"
;
