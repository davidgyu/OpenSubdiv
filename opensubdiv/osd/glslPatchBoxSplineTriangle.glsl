//
//   Copyright 2018 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

//----------------------------------------------------------
// Patches.VertexBoxSplineTriangle
//----------------------------------------------------------
#ifdef OSD_PATCH_VERTEX_BOX_SPLINE_TRIANGLE_SHADER

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
// Patches.TessControlBoxSplineTriangle
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_CONTROL_BOX_SPLINE_TRIANGLE_SHADER

patch out vec4 tessOuterLo, tessOuterHi;

in block {
    ControlVertex v;
    OSD_USER_VARYING_DECLARE
} inpt[];

out block {
    OsdPerPatchVertexBezier v;
    OSD_USER_VARYING_DECLARE
} outpt[15];

layout(vertices = 15) out;

void main()
{
    vec3 cv[12];
    for (int i=0; i<12; ++i) {
        cv[i] = inpt[i].v.position.xyz;
    }

    ivec3 patchParam = OsdGetPatchParam(OsdGetPatchIndex(gl_PrimitiveID));
    OsdComputePerPatchVertexBoxSplineTriangle(
        patchParam, gl_InvocationID, cv, outpt[gl_InvocationID].v);

    OSD_USER_VARYING_PER_CONTROL_POINT(gl_InvocationID, gl_InvocationID);

#if defined OSD_ENABLE_SCREENSPACE_TESSELLATION
    // Wait for all basis conversion to be finished
    barrier();
#endif
    if (gl_InvocationID == 0) {
        vec4 tessLevelOuter = vec4(0);
        vec2 tessLevelInner = vec2(0);

        OSD_PATCH_CULL(12);

#if defined OSD_ENABLE_SCREENSPACE_TESSELLATION
        // Gather bezier control points to compute limit surface tess levels
        vec3 bezcv[15];
        for (int i=0; i<15; ++i) {
            bezcv[i] = outpt[i].v.P;
        }
        OsdEvalPatchBezierTriangleTessLevels(
                         bezcv, patchParam,
                         tessLevelOuter, tessLevelInner,
                         tessOuterLo, tessOuterHi);
#else
        OsdGetTessLevelsUniformTriangle(patchParam,
                        tessLevelOuter, tessLevelInner,
                        tessOuterLo, tessOuterHi);

#endif

        gl_TessLevelOuter[0] = tessLevelOuter[0];
        gl_TessLevelOuter[1] = tessLevelOuter[1];
        gl_TessLevelOuter[2] = tessLevelOuter[2];

        gl_TessLevelInner[0] = tessLevelInner[0];
    }
}

#endif

//----------------------------------------------------------
// Patches.TessEvalBoxSplineTriangle
//----------------------------------------------------------
#ifdef OSD_PATCH_TESS_EVAL_BOX_SPLINE_TRIANGLE_SHADER

layout(triangles) in;
layout(OSD_SPACING) in;

patch in vec4 tessOuterLo, tessOuterHi;

in block {
    OsdPerPatchVertexBezier v;
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

    OsdPerPatchVertexBezier cv[15];
    for (int i = 0; i < 15; ++i) {
        cv[i] = inpt[i].v;
    }

    vec2 UV = OsdGetTessParameterizationTriangle(gl_TessCoord.xyz,
                                                 tessOuterLo,
                                                 tessOuterHi);

    ivec3 patchParam = inpt[0].v.patchParam;
    OsdEvalPatchBezierTriangle(patchParam, UV, cv, P, dPu, dPv, N, dNu, dNv);

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
    outpt.v.patchCoord = OsdInterpolatePatchCoordTriangle(UV, patchParam);

    OSD_USER_VARYING_PER_EVAL_POINT_TRIANGLE(UV, 4, 5, 8);

    OSD_DISPLACEMENT_CALLBACK;

    gl_Position = OsdProjectionMatrix() * outpt.v.position;
}

#endif
