//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../../regression/common/shape_utils.h"
#include "../../regression/shapes/all.h"


static std::vector<ShapeDesc> g_defaultShapes;

//------------------------------------------------------------------------------
static void initShapes() {
    g_defaultShapes.push_back( ShapeDesc("catmark_cube",             catmark_cube,             kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_cube_corner0",     catmark_cube_corner0,     kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_cube_corner1",     catmark_cube_corner1,     kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_cube_corner2",     catmark_cube_corner2,     kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_cube_corner3",     catmark_cube_corner3,     kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_cube_corner4",     catmark_cube_corner4,     kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_cube_creases0",    catmark_cube_creases0,    kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_cube_creases1",    catmark_cube_creases1,    kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_dart_edgecorner",  catmark_dart_edgecorner,  kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_dart_edgeonly",    catmark_dart_edgeonly,    kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_edgecorner",       catmark_edgecorner,       kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_edgeonly",         catmark_edgeonly,         kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_edgenone",         catmark_edgenone,         kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_quadstrips",       catmark_quadstrips,       kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_fan",              catmark_fan,              kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_gregory_test1",    catmark_gregory_test1,    kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_gregory_test2",    catmark_gregory_test2,    kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_gregory_test3",    catmark_gregory_test3,    kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_gregory_test4",    catmark_gregory_test4,    kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_hole_test1",       catmark_hole_test1,       kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_hole_test2",       catmark_hole_test2,       kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_pyramid_creases0", catmark_pyramid_creases0, kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_pyramid_creases1", catmark_pyramid_creases1, kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_pyramid",          catmark_pyramid,          kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_tent_creases0",    catmark_tent_creases0,    kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_tent_creases1",    catmark_tent_creases1 ,   kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_tent",             catmark_tent,             kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_torus",            catmark_torus,            kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_torus_creases0",   catmark_torus_creases0,   kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_bishop",           catmark_bishop,           kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_car",              catmark_car,              kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_helmet",           catmark_helmet,           kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_pawn",             catmark_pawn,             kCatmark ) );
    g_defaultShapes.push_back( ShapeDesc("catmark_rook",             catmark_rook,             kCatmark ) );

    g_defaultShapes.push_back( ShapeDesc("loop_cube",                loop_cube,                kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_cubes_semisharp",     loop_cubes_semisharp,     kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_cubes_infsharp",      loop_cubes_infsharp,      kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_cube_asymmetric",     loop_cube_asymmetric,     kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_triangle_edgecorner", loop_triangle_edgecorner, kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_triangle_edgeonly",   loop_triangle_edgeonly,   kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_triangle_edgenone",   loop_triangle_edgenone,   kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_xord_interior",       loop_xord_interior,       kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_xord_boundary",       loop_xord_boundary,       kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_icosahedron",         loop_icosahedron,         kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_icos_semisharp",      loop_icos_semisharp,      kLoop ) );
    g_defaultShapes.push_back( ShapeDesc("loop_icos_infsharp",       loop_icos_infsharp,       kLoop ) );

    //  REMEMBER:  no Bilinear shapes here -- application requires a tessellation shader
}
//------------------------------------------------------------------------------
