//
//   Copyright 2015 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

//----------------------------------------------------------
// Patches.VertexGregoryBasis
//----------------------------------------------------------
#ifdef OSD_PATCH_VERTEX_GREGORY_BASIS_SHADER

layout(location = 0) in vec4 position;
OSD_USER_VARYING_ATTRIBUTE_DECLARE

out block {
    ControlVertex v;
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
    outpt.v.position = position;
    OSD_PATCH_CULL_COMPUTE_CLIPFLAGS(position);
    OSD_USER_VARYING_PER_VERTEX();
}

#endif

//----------------------------------------------------------
// Patches.TessControlGregoryBasis
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_CONTROL_GREGORY_BASIS_SHADER

patch out vec4 tessOuterLo, tessOuterHi;

in block {
    ControlVertex v;
    OSD_USER_VARYING_DECLARE
} inpt[];

out block {
    OsdPerPatchVertexGregoryBasis v;
    OSD_USER_VARYING_DECLARE
} outpt[20];

layout(vertices = 20) out;

void main()
{
    vec3 cv = inpt[gl_InvocationID].v.position.xyz;

    ivec3 patchParam = OsdGetPatchParam(OsdGetPatchIndex(gl_PrimitiveID));
    OsdComputePerPatchVertexGregoryBasis(
        patchParam, gl_InvocationID, cv, outpt[gl_InvocationID].v);

    OSD_USER_VARYING_PER_CONTROL_POINT(gl_InvocationID, gl_InvocationID);

    if (gl_InvocationID == 0) {
        vec4 tessLevelOuter = vec4(0);
        vec2 tessLevelInner = vec2(0);

        OSD_PATCH_CULL(20);

#if defined OSD_ENABLE_SCREENSPACE_TESSELLATION
        // Gather bezier control points to compute limit surface tess levels
        OsdPerPatchVertexBezier bezcv[16];
        bezcv[ 0].P = inpt[ 0].v.position.xyz;
        bezcv[ 1].P = inpt[ 1].v.position.xyz;
        bezcv[ 2].P = inpt[ 7].v.position.xyz;
        bezcv[ 3].P = inpt[ 5].v.position.xyz;
        bezcv[ 4].P = inpt[ 2].v.position.xyz;
        bezcv[ 5].P = inpt[ 3].v.position.xyz;
        bezcv[ 6].P = inpt[ 8].v.position.xyz;
        bezcv[ 7].P = inpt[ 6].v.position.xyz;
        bezcv[ 8].P = inpt[16].v.position.xyz;
        bezcv[ 9].P = inpt[18].v.position.xyz;
        bezcv[10].P = inpt[13].v.position.xyz;
        bezcv[11].P = inpt[12].v.position.xyz;
        bezcv[12].P = inpt[15].v.position.xyz;
        bezcv[13].P = inpt[17].v.position.xyz;
        bezcv[14].P = inpt[11].v.position.xyz;
        bezcv[15].P = inpt[10].v.position.xyz;

        OsdEvalPatchBezierTessLevels(
                bezcv, patchParam,
                tessLevelOuter, tessLevelInner,
                tessOuterLo, tessOuterHi);
#else
        OsdGetTessLevelsUniform(
                patchParam,
                tessLevelOuter, tessLevelInner,
                tessOuterLo, tessOuterHi);
#endif

        gl_TessLevelOuter[0] = tessLevelOuter[0];
        gl_TessLevelOuter[1] = tessLevelOuter[1];
        gl_TessLevelOuter[2] = tessLevelOuter[2];
        gl_TessLevelOuter[3] = tessLevelOuter[3];

        gl_TessLevelInner[0] = tessLevelInner[0];
        gl_TessLevelInner[1] = tessLevelInner[1];
    }
}

#endif

//----------------------------------------------------------
// Patches.TessEvalGregoryBasis
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_EVAL_GREGORY_BASIS_SHADER

layout(quads) in;
layout(OSD_SPACING) in;

patch in vec4 tessOuterLo, tessOuterHi;

in block {
    OsdPerPatchVertexGregoryBasis v;
    OSD_USER_VARYING_DECLARE
} inpt[];

out block {
    OutputVertex v;
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
    vec3 P = vec3(0), dPu = vec3(0), dPv = vec3(0);
    vec3 N = vec3(0), dNu = vec3(0), dNv = vec3(0);

    vec3 cv[20];
    for (int i = 0; i < 20; ++i) {
        cv[i] = inpt[i].v.P;
    }

    vec2 UV = OsdGetTessParameterization(gl_TessCoord.xy,
                                         tessOuterLo,
                                         tessOuterHi);

    ivec3 patchParam = inpt[0].v.patchParam;
    OsdEvalPatchGregory(patchParam, UV, cv, P, dPu, dPv, N, dNu, dNv);

    // all code below here is client code
    outpt.v.position = OsdModelViewMatrix() * vec4(P, 1.0f);
    outpt.v.normal = (OsdModelViewMatrix() * vec4(N, 0.0f)).xyz;
    outpt.v.tangent = (OsdModelViewMatrix() * vec4(dPu, 0.0f)).xyz;
    outpt.v.bitangent = (OsdModelViewMatrix() * vec4(dPv, 0.0f)).xyz;
#ifdef OSD_COMPUTE_NORMAL_DERIVATIVES
    outpt.v.Nu = dNu;
    outpt.v.Nv = dNv;
#endif

    outpt.v.tessCoord = UV;
    outpt.v.patchCoord = OsdInterpolatePatchCoord(UV, patchParam);

    OSD_USER_VARYING_PER_EVAL_POINT(UV, 0, 5, 15, 10);

    OSD_DISPLACEMENT_CALLBACK;

    gl_Position = OsdProjectionMatrix() * outpt.v.position;
}

#endif
