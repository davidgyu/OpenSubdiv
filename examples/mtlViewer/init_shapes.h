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
    g_defaultShapes.push_back(ShapeDesc("catmark_toroidal_tet",     catmark_toroidal_tet,     kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_cube",             catmark_cube,             kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_cubes_semisharp",  catmark_cubes_semisharp,  kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_cubes_infsharp",   catmark_cubes_infsharp,   kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_pyramid",          catmark_pyramid,          kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_pyramid_creases0", catmark_pyramid_creases0, kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_pyramid_creases1", catmark_pyramid_creases1, kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_torus",            catmark_torus,            kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_torus_creases0",   catmark_torus_creases0,   kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_torus_creases1",   catmark_torus_creases1,   kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_edgecorner",       catmark_edgecorner,       kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_edgeonly",         catmark_edgeonly,         kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_edgenone",         catmark_edgenone,         kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_quadstrips",       catmark_quadstrips,       kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_xord_interior",    catmark_xord_interior,    kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_xord_boundary",    catmark_xord_boundary,    kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_val2_interior",    catmark_val2_interior,    kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_nonquads",         catmark_nonquads,         kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_single_crease",    catmark_single_crease,    kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_inf_crease0",      catmark_inf_crease0,      kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_inf_crease1",      catmark_inf_crease1,      kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_nonman_verts",     catmark_nonman_verts,     kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_nonman_edges",     catmark_nonman_edges,     kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_hole_test1",       catmark_hole_test1,       kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_hole_test2",       catmark_hole_test2,       kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_hole_test3",       catmark_hole_test3,       kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_hole_test4",       catmark_hole_test4,       kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_chaikin0",         catmark_chaikin0,         kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_chaikin1",         catmark_chaikin1,         kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_chaikin2",         catmark_chaikin2,         kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_smoothtris0",      catmark_smoothtris0,      kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_smoothtris1",      catmark_smoothtris1,      kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_pole8",            catmark_pole8,            kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_pole64",           catmark_pole64,           kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_nonman_quadpole64",catmark_nonman_quadpole64,kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_nonman_edge100",   catmark_nonman_edge100,   kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_helmet",           catmark_helmet,           kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_bishop",           catmark_bishop,           kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_pawn",             catmark_pawn,             kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_rook",             catmark_rook,             kCatmark));
    g_defaultShapes.push_back(ShapeDesc("catmark_car",              catmark_car,              kCatmark));

    g_defaultShapes.push_back(ShapeDesc("loop_toroidal_tet",        loop_toroidal_tet,        kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_tetrahedron",         loop_tetrahedron,         kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_cube",                loop_cube,                kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_cubes_semisharp",     loop_cubes_semisharp,     kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_cubes_infsharp",      loop_cubes_infsharp,      kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_cube_asymmetric",     loop_cube_asymmetric,     kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_icosahedron",         loop_icosahedron,         kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_icos_semisharp",      loop_icos_semisharp,      kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_icos_infsharp",       loop_icos_infsharp,       kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_triangle_edgecorner", loop_triangle_edgecorner, kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_triangle_edgeonly",   loop_triangle_edgeonly,   kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_triangle_edgenone",   loop_triangle_edgenone,   kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_xord_interior",       loop_xord_interior,       kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_xord_boundary",       loop_xord_boundary,       kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_nonman_verts",        loop_nonman_verts,        kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_nonman_edges",        loop_nonman_edges,        kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_pole64",              loop_pole64,              kLoop));
    g_defaultShapes.push_back(ShapeDesc("loop_nonman_edge100",      loop_nonman_edge100,      kLoop));

    g_defaultShapes.push_back(ShapeDesc("bilinear_cube",            bilinear_cube,            kBilinear));
    g_defaultShapes.push_back(ShapeDesc("bilinear_nonplanar",       bilinear_nonplanar,       kBilinear));
    g_defaultShapes.push_back(ShapeDesc("bilinear_nonquads0",       bilinear_nonquads0,       kBilinear));
    g_defaultShapes.push_back(ShapeDesc("bilinear_nonquads1",       bilinear_nonquads1,       kBilinear));
}
//------------------------------------------------------------------------------
