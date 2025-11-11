//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

#include "far_utils.h"

struct FVarVertex {

    float u,v;

    void Clear() {
        u=v=0.0f;
    }

    void AddWithWeight(FVarVertex const & src, float weight) {
        u += weight * src.u;
        v += weight * src.v;
    }
};

void
InterpolateFVarData(OpenSubdiv::Far::TopologyRefiner & refiner,
    Shape const & shape, std::vector<float> & fvarData) {

    int channel = 0,    // shapes only have 1 UV channel
        fvarWidth = 2;

    int maxlevel = refiner.GetMaxLevel(),
        numValuesM = refiner.GetLevel(maxlevel).GetNumFVarValues(channel),
        numValuesTotal = refiner.GetNumFVarValuesTotal(channel);

    if (shape.uvs.empty() || numValuesTotal<=0) {
        return;
    }

    OpenSubdiv::Far::PrimvarRefiner primvarRefiner(refiner);

    if (refiner.IsUniform()) {

        // For uniform we only keep the highest level of refinement:
        fvarData.resize(numValuesM * fvarWidth);

        std::vector<FVarVertex> buffer(numValuesTotal - numValuesM);

        FVarVertex * src = &buffer[0];
        memcpy(src, &shape.uvs[0], shape.uvs.size()*sizeof(float));

        //  Defer the last level to treat separately with its alternate destination:
        for (int level = 1; level < maxlevel; ++level) {
            FVarVertex * dst = src + refiner.GetLevel(level-1).GetNumFVarValues(channel);

            primvarRefiner.InterpolateFaceVarying(level, src, dst, channel);

            src = dst;
        }

        FVarVertex * dst = reinterpret_cast<FVarVertex *>(&fvarData[0]);
        primvarRefiner.InterpolateFaceVarying(maxlevel, src, dst, channel);

    } else {

        // For adaptive we keep all levels:
        fvarData.resize(numValuesTotal * fvarWidth);

        FVarVertex * src = reinterpret_cast<FVarVertex *>(&fvarData[0]);
        memcpy(src, &shape.uvs[0], shape.uvs.size()*sizeof(float));

        for (int level = 1; level <= maxlevel; ++level) {
            FVarVertex * dst = src + refiner.GetLevel(level-1).GetNumFVarValues(channel);

            primvarRefiner.InterpolateFaceVarying(level, src, dst, channel);

            src = dst;
        }
    }
}
