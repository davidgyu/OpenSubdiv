//
//   Copyright 2019 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "../common/viewerArgsUtils.h"

#include "../../regression/common/arg_utils.h"
#include "../../regression/common/shape_utils.h"
#include "../common/objAnim.h"

#include <stdio.h>

namespace ViewerArgsUtils {

const ObjAnim *
PopulateAnimShapes(const ArgOptions &args, 
                   std::vector<ShapeDesc> *defaultShapes)
{
    if (args.GetObjFiles().empty())
        return NULL;

    const ObjAnim *objAnim = ObjAnim::Create(args.GetObjFiles(),
        args.GetDefaultScheme());

    if (objAnim && defaultShapes) {
        defaultShapes->push_back(ShapeDesc(args.GetObjFiles()[0], "", 
            args.GetDefaultScheme()));
    }

    return objAnim;

}

void 
PopulateShapes(const ArgOptions &args,
               std::vector<ShapeDesc> *defaultShapes)
{
    if (defaultShapes) {
        args.AppendObjShapes(*defaultShapes, true /* print warnings */);
    }
}

void 
PopulateShapesOrAnimShapes(
    const ArgOptions &args, 
    std::vector<ShapeDesc> *defaultShapes, 
    const ObjAnim **objAnim)
{
    if (!defaultShapes)
        return;

    if (args.GetObjFiles().empty())
        return;

    if (args.GetObjsAreAnim()) {

        if (!objAnim) {
            printf("Warning: animations of objs are unsupported in this "
                   "viewer.\n");
            return;
        }

        *objAnim = PopulateAnimShapes(args, defaultShapes);

    } else {

        PopulateShapes(args, defaultShapes);

    }
}

}
