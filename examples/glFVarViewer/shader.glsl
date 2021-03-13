//
//   Copyright 2013 Pixar
//
//   Licensed under the Apache License, Version 2.0 (the "Apache License")
//   with the following modification; you may not use this file except in
//   compliance with the Apache License and the following modification to it:
//   Section 6. Trademarks. is deleted and replaced with:
//
//   6. Trademarks. This License does not grant permission to use the trade
//      names, trademarks, service marks, or product names of the Licensor
//      and its affiliates, except as required to comply with Section 4(c) of
//      the License and to reproduce the content of the NOTICE file.
//
//   You may obtain a copy of the Apache License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the Apache License with the above modification is
//   distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
//   KIND, either express or implied. See the Apache License for the specific
//   language governing permissions and limitations under the Apache License.
//

#undef OSD_USER_VARYING_DECLARE
#define OSD_USER_VARYING_DECLARE \
    vec3 color;

#undef OSD_USER_VARYING_ATTRIBUTE_DECLARE
#define OSD_USER_VARYING_ATTRIBUTE_DECLARE \
    layout(location = 1) in vec3 color;

#undef OSD_USER_VARYING_PER_VERTEX
#define OSD_USER_VARYING_PER_VERTEX() \
    outpt.color = color

#undef OSD_USER_VARYING_PER_CONTROL_POINT
#define OSD_USER_VARYING_PER_CONTROL_POINT(ID_OUT, ID_IN) \
    outpt[ID_OUT].color = inpt[ID_IN].color

#undef OSD_USER_VARYING_PER_EVAL_POINT
#define OSD_USER_VARYING_PER_EVAL_POINT(UV, a, b, c, d) \
    outpt.color = \
        mix(mix(inpt[a].color, inpt[b].color, UV.x), \
            mix(inpt[c].color, inpt[d].color, UV.x), UV.y)

//--------------------------------------------------------------
// Uniform Blocks
//--------------------------------------------------------------

layout(std140) uniform Transform {
    mat4 ModelViewMatrix;
    mat4 ProjectionMatrix;
    mat4 ModelViewProjectionMatrix;
    mat4 ModelViewInverseMatrix;
    mat4 UvViewMatrix;
};

layout(std140) uniform Tessellation {
    float TessLevel;
};

uniform int GregoryQuadOffsetBase;
uniform int PrimitiveIdBase;

//--------------------------------------------------------------
// Osd external functions
//--------------------------------------------------------------

mat4 OsdModelViewMatrix()
{
    return ModelViewMatrix;
}
mat4 OsdProjectionMatrix()
{
    return ProjectionMatrix;
}
mat4 OsdModelViewProjectionMatrix()
{
    return ModelViewProjectionMatrix;
}
float OsdTessLevel()
{
    return TessLevel;
}
int OsdGregoryQuadOffsetBase()
{
    return GregoryQuadOffsetBase;
}
int OsdPrimitiveIdBase()
{
    return PrimitiveIdBase;
}
int OsdBaseVertex()
{
    return 0;
}

//--------------------------------------------------------------
// Vertex Shader
//--------------------------------------------------------------
#ifdef VERTEX_SHADER

layout (location=0) in vec4 position;
OSD_USER_VARYING_ATTRIBUTE_DECLARE

out block {
    OutputVertex v;
    OSD_USER_VARYING_DECLARE
} outpt;

void main()
{
    outpt.v.position = ModelViewMatrix * position;

    // We don't actually want to write all these, but some
    // compilers complain during about failing to fully write
    // outpt.v if they are excluded.
    outpt.v.normal = vec3(0);
    outpt.v.tangent = vec3(0);
    outpt.v.bitangent = vec3(0);
    outpt.v.patchCoord = vec4(0);
    outpt.v.tessCoord = vec2(0);
#if defined OSD_COMPUTE_NORMAL_DERIVATIVES
    outpt.v.Nu = vec3(0);
    outpt.v.Nv = vec3(0);
#endif
    // --

    OSD_USER_VARYING_PER_VERTEX();
}

#endif

//--------------------------------------------------------------
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

#ifdef PRIM_QUAD

    layout(lines_adjacency) in;

    #define EDGE_VERTS 4

#endif // PRIM_QUAD

#ifdef  PRIM_TRI

    layout(triangles) in;

    #define EDGE_VERTS 3

#endif // PRIM_TRI


layout(triangle_strip, max_vertices = EDGE_VERTS) out;
in block {
    OutputVertex v;
    OSD_USER_VARYING_DECLARE
} inpt[EDGE_VERTS];

out block {
    OutputVertex v;
    noperspective out vec3 barycentric;
    OSD_USER_VARYING_DECLARE
} outpt;

uniform isamplerBuffer OsdFVarParamBuffer;
layout(std140) uniform OsdFVarArrayData {
    OsdPatchArray fvarPatchArray[2];
};

vec2
interpolateFaceVarying(vec2 uv, int fvarOffset)
{
    int patchIndex = OsdGetPatchIndex(gl_PrimitiveID);

    OsdPatchArray array = fvarPatchArray[0];

    ivec3 fvarPatchParam = texelFetch(OsdFVarParamBuffer, patchIndex).xyz;
    OsdPatchParam param = OsdPatchParamInit(fvarPatchParam.x,
                                            fvarPatchParam.y,
                                            fvarPatchParam.z);

    int patchType = OsdPatchParamIsRegular(param) ? array.regDesc : array.desc;

    float wP[20], wDu[20], wDv[20], wDuu[20], wDuv[20], wDvv[20];
    int numPoints = OsdEvaluatePatchBasisNormalized(patchType, param,
                uv.s, uv.t, wP, wDu, wDv, wDuu, wDuv, wDvv);

    int primOffset = patchIndex * array.stride;

    vec2 result = vec2(0);
    for (int i=0; i<numPoints; ++i) {
        int index = (primOffset+i)*OSD_FVAR_WIDTH + fvarOffset;
        vec2 cv = vec2(texelFetch(OsdFVarDataBuffer, index).s,
                       texelFetch(OsdFVarDataBuffer, index + 1).s);
        result += wP[i] * cv;
    }

    return result;
}

void emit(int index, vec3 normal)
{
    outpt.v.position = inpt[index].v.position;
#ifdef SMOOTH_NORMALS
    outpt.v.normal = inpt[index].v.normal;
#else
    outpt.v.normal = normal;
#endif

#ifdef SHADING_FACEVARYING_UNIFORM_SUBDIVISION
    // interpolate fvar data at refined tri or quad vertex locations
#ifdef PRIM_TRI
    vec2 trist[3] = vec2[](vec2(0,0), vec2(1,0), vec2(0,1));
    vec2 st = trist[index];
#endif
#ifdef PRIM_QUAD
    vec2 quadst[4] = vec2[](vec2(0,0), vec2(1,0), vec2(1,1), vec2(0,1));
    vec2 st = quadst[index];
#endif
#else
    // interpolate fvar data at tessellated vertex locations
    vec2 st = inpt[index].v.tessCoord;
#endif

    vec2 uv = interpolateFaceVarying(st, /*fvarOffset*/0);

    outpt.color = vec3(uv.s, uv.t, 0);

#ifdef GEOMETRY_UV_VIEW
    uv = 2 * uv - vec2(1);
    gl_Position = UvViewMatrix * vec4(uv.s, uv.y, 0, 1);
#else
    gl_Position = ProjectionMatrix * inpt[index].v.position;
#endif

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    vec3 coords[3] = vec3[](vec3(0,0,1),vec3(1,0,0),vec3(0,1,0));
    outpt.barycentric = coords[index];
#endif
#ifdef PRIM_QUAD
    vec3 coords[4] = vec3[](vec3(0,0,1),vec3(1,0,0),vec3(0,0,1),vec3(0,1,0));
    outpt.barycentric = coords[index];
#endif
#else
    outpt.barycentric = vec3(0);
#endif

    EmitVertex();
}

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

#ifdef PRIM_QUAD
    vec3 A = (inpt[0].v.position - inpt[1].v.position).xyz;
    vec3 B = (inpt[3].v.position - inpt[1].v.position).xyz;
    vec3 C = (inpt[2].v.position - inpt[1].v.position).xyz;
    vec3 n0 = normalize(cross(B, A));

    emit(0, n0);
    emit(1, n0);
    emit(3, n0);
    emit(2, n0);
#endif // PRIM_QUAD

#ifdef PRIM_TRI
    vec3 A = (inpt[1].v.position - inpt[0].v.position).xyz;
    vec3 B = (inpt[2].v.position - inpt[0].v.position).xyz;
    vec3 n0 = normalize(cross(B, A));

    emit(0, n0);
    emit(1, n0);
    emit(2, n0);
#endif // PRIM_TRI

    EndPrimitive();
}

#endif

//--------------------------------------------------------------
// Fragment Shader
//--------------------------------------------------------------
#ifdef FRAGMENT_SHADER

in block {
    OutputVertex v;
    noperspective in vec3 barycentric;
    OSD_USER_VARYING_DECLARE
} inpt;

out vec4 outColor;

vec4
edgeColor(vec4 Cfill, vec3 barycentric)
{
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    vec3 dist = max(vec3(0), barycentric / fwidth(barycentric));
#endif
#ifdef PRIM_QUAD
    vec3 dist = max(vec3(0), barycentric / fwidth(vec3(barycentric.xy,1)));
#endif

    float d = min(dist.x, min(dist.y, dist.z));
    float p = exp2(-2 * d * d);

    vec4 Cedge = vec4(1.0, 1.0, 0.0, 1.0);

#if defined(GEOMETRY_OUT_WIRE)
    if (p < 0.25) discard;
#endif

    Cfill.rgb = mix(Cfill.rgb, Cedge.rgb, p);
#endif

    return Cfill;
}

#if defined(PRIM_QUAD) || defined(PRIM_TRI)
void
main()
{
#ifdef GEOMETRY_UV_VIEW
    outColor = edgeColor(vec4(0.9), vec3(0));
    return;

#else
    vec3 N = (gl_FrontFacing ? inpt.v.normal : -inpt.v.normal);

    // generating a checkerboard pattern
    int checker = int(floor(20*inpt.color.r)+floor(20*inpt.color.g))&1;
    vec4 color = vec4(inpt.color.rg*checker, 1-checker, 1);

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    color = edgeColor(color, inpt.barycentric);
#endif
    outColor = color;
#endif
}
#endif

#endif
