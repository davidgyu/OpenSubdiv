#line 0 "osd/mtlPatchBSpline.metal"

//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

//----------------------------------------------------------
// Patches.BSpline.Hull
//----------------------------------------------------------

void OsdComputePerVertex(
        float4 position,
        threadgroup HullVertex& hullVertex,
        int vertexId,
        float4x4 modelViewProjectionMatrix,
        OsdPatchParamBufferSet osdBuffers
        )
{
    hullVertex.position = position;
#if OSD_ENABLE_PATCH_CULL
    float4 clipPos = mul(modelViewProjectionMatrix, position);
    short3 clip0 = short3(clipPos.x < clipPos.w,
                          clipPos.y < clipPos.w,
                          clipPos.z < clipPos.w);
    short3 clip1 = short3(clipPos.x > -clipPos.w,
                          clipPos.y > -clipPos.w,
                          clipPos.z > -clipPos.w);
    hullVertex.clipFlag = short3(clip0) + 2*short3(clip1);
#endif
}

//----------------------------------------------------------
// Patches.BSpline.Factors
//----------------------------------------------------------

void OsdComputePerPatchBSplineFactors(
        int3 patchParam,
        float tessLevel,
        float4x4 projectionMatrix,
        float4x4 modelViewMatrix,
        device OsdPerPatchVertexBezier* patch,
#if !USE_PTVS_FACTORS
        device OsdPerPatchTessFactors& patchFactors,
#endif
        device MTLQuadTessellationFactorsHalf& quadFactors
        )
{
    float4 tessLevelOuter = float4(0);
    float2 tessLevelInner = float2(0);
    float4 tessOuterLo = float4(0);
    float4 tessOuterHi = float4(0);

#if OSD_ENABLE_SCREENSPACE_TESSELLATION
    OsdEvalPatchBezierTessLevels(
        tessLevel,
        projectionMatrix,
        modelViewMatrix,
        patch,
        patchParam,
        tessLevelOuter,
        tessLevelInner,
        tessOuterLo,
        tessOuterHi
        );
#else
    OsdGetTessLevelsUniform(
        tessLevel,
        patchParam,
        tessLevelOuter,
        tessLevelInner,
        tessOuterLo,
        tessOuterHi
        );
#endif

    quadFactors.edgeTessellationFactor[0] = tessLevelOuter[0];
    quadFactors.edgeTessellationFactor[1] = tessLevelOuter[1];
    quadFactors.edgeTessellationFactor[2] = tessLevelOuter[2];
    quadFactors.edgeTessellationFactor[3] = tessLevelOuter[3];
    quadFactors.insideTessellationFactor[0] = tessLevelInner[0];
    quadFactors.insideTessellationFactor[1] = tessLevelInner[1];
#if !USE_PTVS_FACTORS
    patchFactors.tessOuterLo = tessOuterLo;
    patchFactors.tessOuterHi = tessOuterHi;
#endif
}

void OsdComputePerPatchFactors(
        int3 patchParam,
        float tessLevel,
        unsigned patchID,
        float4x4 projectionMatrix,
        float4x4 modelViewMatrix,
        OsdPatchParamBufferSet osdBuffer,
        threadgroup PatchVertexType* patchVertices,
        device MTLQuadTessellationFactorsHalf& quadFactors
        )
{
    OsdComputePerPatchBSplineFactors(
        patchParam,
        tessLevel,
        projectionMatrix,
        modelViewMatrix,
        osdBuffer.perPatchVertexBuffer + patchID * VERTEX_CONTROL_POINTS_PER_PATCH,
#if !USE_PTVS_FACTORS
        osdBuffer.patchTessBuffer[patchID],
#endif
        quadFactors
        );
}

//----------------------------------------------------------
// Patches.BSpline.Vertex
//----------------------------------------------------------

void OsdComputePerPatchVertex(
        int3 patchParam,
        unsigned ID,
        unsigned PrimitiveID,
        unsigned ControlID,
        threadgroup PatchVertexType* patchVertices,
        OsdPatchParamBufferSet osdBuffers
        )
{
    OsdComputePerPatchVertexBSpline(patchParam, ID,
        patchVertices, osdBuffers.perPatchVertexBuffer[ControlID]);
}

//----------------------------------------------------------
// Patches.BSpline.Domain
//----------------------------------------------------------

template<typename PerPatchVertexBezier>
OsdPatchVertex ds_regular_patches(
        const float tessLevel,
#if !USE_PTVS_FACTORS
        float4 tessOuterLo,
        float4 tessOuterHi,
#endif
        PerPatchVertexBezier cv,
        int3 patchParam,
        float2 domainCoord
        )
{
#if USE_PTVS_FACTORS
    float4 tessOuterLo(0), tessOuterHi(0);
    OsdGetTessLevelsUniform(tessLevel, patchParam, tessOuterLo, tessOuterHi);
#endif

    float2 UV = OsdGetTessParameterization(domainCoord,
                                           tessOuterLo,
                                           tessOuterHi);

    OsdPatchVertex output;

    float3 P = float3(0), dPu = float3(0), dPv = float3(0);
    float3 N = float3(0), dNu = float3(0), dNv = float3(0);
    float2 vSegments = float2(0);

    OsdEvalPatchBezier(patchParam, UV, cv, P, dPu, dPv, N, dNu, dNv, vSegments);

    output.normal = N;
    output.tangent = dPu;
    output.bitangent = dPv;
#if OSD_COMPUTE_NORMAL_DERIVATIVES
    output.Nu = dNu;
    output.Nv = dNv;
#endif
#if OSD_PATCH_ENABLE_SINGLE_CREASE
    output.vSegments = vSegments;
#endif

    output.tessCoord = UV;
    output.patchCoord = OsdInterpolatePatchCoord(UV, patchParam);
    output.position = P;
    return output;
}

#if USE_STAGE_IN
template<typename PerPatchVertexBezier>
#endif
OsdPatchVertex OsdComputePatch(
        float tessLevel,
        float2 domainCoord,
        unsigned patchID,
#if USE_STAGE_IN
        PerPatchVertexBezier osdPatch
#else
        OsdVertexBufferSet osdBuffers
#endif
        )
{
    return ds_regular_patches(
            tessLevel,
#if !USE_PTVS_FACTORS
#if USE_STAGE_IN
            osdPatch.tessOuterLo,
            osdPatch.tessOuterHi,
#else
            osdBuffers.patchTessBuffer[patchID].tessOuterLo,
            osdBuffers.patchTessBuffer[patchID].tessOuterHi,
#endif
#endif
#if USE_STAGE_IN
            osdPatch.cv,
            osdPatch.patchParam,
#else
            osdBuffers.perPatchVertexBuffer + patchID * VERTEX_CONTROL_POINTS_PER_PATCH,
            osdBuffers.patchParamBuffer[patchID],
#endif
            domainCoord
            );
}

