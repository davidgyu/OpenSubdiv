//
//   Copyright 2019 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef VIEWER_ARGS_UTILS_H
#define VIEWER_ARGS_UTILS_H

#include <vector>

class ArgOptions;
struct ShapeDesc;
class ObjAnim;

namespace ViewerArgsUtils {

// From the list of obj files in args, populates the
// defaultShapes vector, treating the objs as an animated series, returning
// an ObjAnim object.
const ObjAnim *PopulateAnimShapes(const ArgOptions &args, 
                                  std::vector<ShapeDesc> *defaultShapes = 0);

// From the list of obj files in args, populates the
// defaultShapes vector.
void PopulateShapes(const ArgOptions &args,
                    std::vector<ShapeDesc> *defaultShapes);

// From the list of obj files in args, populates the
// defaultShapes vector and the objAnim object if animated objs have
// been specified in args.
void PopulateShapesOrAnimShapes(
                    const ArgOptions &args,
                    std::vector<ShapeDesc> *defaultShapes, 
                    const ObjAnim **objAnim);


} // namespace ViewerArgsUtils

#endif // VIEWER_ARGS_UTILS_H
