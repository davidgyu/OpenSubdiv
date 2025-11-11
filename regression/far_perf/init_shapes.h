//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../common/shape_utils.h"
#include "../shapes/all.h"


static std::vector<ShapeDesc> g_shapes;

//------------------------------------------------------------------------------
static void initShapes() {
    g_shapes.push_back( ShapeDesc("catmark_car",     catmark_car,   kCatmark ) );
    g_shapes.push_back( ShapeDesc("catmark_pole64", catmark_pole64, kCatmark ) );
}
//------------------------------------------------------------------------------
