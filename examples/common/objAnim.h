//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#ifndef OBJ_ANIM_H
#define OBJ_ANIM_H

#include "../../regression/common/shape_utils.h"

#include <vector>

class ObjAnim {

public:

    // Factory function
    static ObjAnim const * Create(std::vector<char const *> objFiles,
                                  Scheme scheme, bool isLeftHanded=false);

    // Destructor
    ~ObjAnim();

    // Populates 'positions' with the interpolated vertex data for a given
    // time.
    void InterpolatePositions(float time, float * positions, int stride) const;

    // Number of key-frames in the animation
    int GetNumKeyframes() const {
        return (int)_positions.size();
    }

    // Returns the full 'Shape'
    Shape const * GetShape() const {
        return _shape;
    }


private:

    ObjAnim();

    Shape const * _shape;

    std::vector<std::vector<float> > _positions;
};

#endif // OBJ_ANIM_H
