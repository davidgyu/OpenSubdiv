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

vec4 PTexLookup(vec4 patchCoord,
                sampler2DArray data,
                samplerBuffer packings,
                isamplerBuffer pages)
{
    vec2 uv = patchCoord.xy;
    int faceID = int(patchCoord.w);
    int page = texelFetch(pages, faceID).x;
    vec4 packing = texelFetch(packings, faceID);
    vec3 coords = vec3( packing.x + uv.x * packing.z,
                        packing.y + uv.y * packing.w,
                        page);

    return texture(data, coords);
}

uniform sampler2DArray textureImage_Data;
uniform samplerBuffer textureImage_Packing;
uniform isamplerBuffer textureImage_Pages;

vec4 displacement(vec4 position, vec3 normal, vec4 patchCoord)
{
    float disp = PTexLookup(patchCoord,
                            textureImage_Data,
                            textureImage_Packing,
                            textureImage_Pages).x;
    return position + 0.01*vec4(disp * normal, 0);
}

//--------------------------------------------------------------
// Uniform Blocks
//--------------------------------------------------------------

layout(std140) uniform Transform {
    mat4 ModelViewMatrix;
    mat4 ProjectionMatrix;
    mat4 ModelViewProjectionMatrix;
    mat4 ModelViewInverseMatrix;
    mat4 ProjectionWithoutPickMatrix;
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
// Geometry Shader
//--------------------------------------------------------------
#ifdef GEOMETRY_SHADER

    layout(triangles) in;

    layout(triangle_strip, max_vertices = 3) out;

    #define EDGE_VERTS 3

    in block {
        OutputVertex v;
    } inpt[3];

out block {
    OutputVertex v;
    noperspective out vec3 barycentric;
} outpt;

void emit(int index, vec4 position, vec3 normal, vec4 patchCoord)
{
    outpt.v.position = position;
    outpt.v.patchCoord = patchCoord;
    outpt.v.normal = normal;

    gl_Position = ProjectionMatrix * outpt.v.position;

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

// --------------------------------------

void main()
{
    gl_PrimitiveID = gl_PrimitiveIDIn;

    vec4 position[3];
    vec4 patchCoord[3];
    vec3 normal[3];

    // patch coords are computed in tessellation shader
    patchCoord[0] = inpt[0].v.patchCoord;
    patchCoord[1] = inpt[1].v.patchCoord;
    patchCoord[2] = inpt[2].v.patchCoord;

#ifdef USE_PTEX_DISPLACEMENT
    position[0] = displacement(inpt[0].v.position, inpt[0].v.normal, patchCoord[0]);
    position[1] = displacement(inpt[1].v.position, inpt[1].v.normal, patchCoord[1]);
    position[2] = displacement(inpt[2].v.position, inpt[2].v.normal, patchCoord[2]);
#else
    position[0] = inpt[0].v.position;
    position[1] = inpt[1].v.position;
    position[2] = inpt[2].v.position;
#endif

    normal[0] = inpt[0].v.normal;
    normal[1] = inpt[1].v.normal;
    normal[2] = inpt[2].v.normal;

    emit(0, position[0], normal[0], patchCoord[0]);
    emit(1, position[1], normal[1], patchCoord[1]);
    emit(2, position[2], normal[2], patchCoord[2]);

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
} inpt;

out vec4 outColor;

#define NUM_LIGHTS 2

struct LightSource {
    vec4 position;
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
};

layout(std140) uniform Lighting {
    LightSource lightSource[NUM_LIGHTS];
};

uniform vec4 diffuseColor = vec4(1);
uniform vec4 ambientColor = vec4(1);

vec4
lighting(vec4 diffuse, vec3 Peye, vec3 Neye)
{
    vec4 color = vec4(0);

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        vec4 Plight = lightSource[i].position;

        vec3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        vec3 n = normalize(Neye);
        vec3 h = normalize(l + vec3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 500.0f);

        color += lightSource[i].ambient * ambientColor
            + d * lightSource[i].diffuse * diffuse
            + s * lightSource[i].specular;
    }

    color.a = 1;
    return color;
}

#ifdef USE_PTEX_DISPLACEMENT
vec3
perturbNormalFromDisplacement(vec3 position, vec3 normal, vec4 patchCoord)
{
    // by Morten S. Mikkelsen
    // http://jbit.net/~sparky/sfgrad_bump/mm_sfgrad_bump.pdf
    // slightly modified for ptex guttering

    vec3 vSigmaS = dFdx(position);
    vec3 vSigmaT = dFdy(position);
    vec3 vN = normal;
    vec3 vR1 = cross(vSigmaT, vN);
    vec3 vR2 = cross(vN, vSigmaS);
    float fDet = dot(vSigmaS, vR1);

    vec2 texDx = dFdx(patchCoord.xy);
    vec2 texDy = dFdy(patchCoord.xy);
    
    // limit forward differencing to the width of ptex gutter
    const float resolution = 128.0;
    float d = (0.5/resolution)/max(length(texDx), length(texDy));
    
    vec4 STll = patchCoord;
    vec4 STlr = patchCoord + d * vec4(texDx.x, texDx.y, 0, 0);
    vec4 STul = patchCoord + d * vec4(texDy.x, texDy.y, 0, 0);
    float Hll = PTexLookup(STll, textureImage_Data, textureImage_Packing, textureImage_Pages).x;
    float Hlr = PTexLookup(STlr, textureImage_Data, textureImage_Packing, textureImage_Pages).x;
    float Hul = PTexLookup(STul, textureImage_Data, textureImage_Packing, textureImage_Pages).x;
    float dBs = (Hlr - Hll)/d;
    float dBt = (Hul - Hll)/d;
    
    vec3 vSurfGrad = sign(fDet) * (dBs * vR1 + dBt * vR2);
    return normalize(abs(fDet) * vN - vSurfGrad);
}
#endif // USE_PTEX_NORMAL

vec4
edgeColor(vec4 Cfill, vec3 barycentric)
{
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    vec3 dist = max(vec3(0),barycentric / fwidth(barycentric));
#endif
#ifdef PRIM_QUAD
    vec3 dist = max(vec3(0),barycentric / fwidth(vec3(barycentric.xy,1)));
#endif

    float d = min(dist.x, min(dist.y, dist.z));
    float p = exp2(-2 * d * d);

#if defined(GEOMETRY_OUT_WIRE)
    if (p < 0.25) discard;
#endif

    vec4 Cedge = vec4(0.5, 0.5, 0.5, 1.0);

    Cfill.rgb = mix(Cfill.rgb, Cedge.rgb, p);
#endif
    return Cfill;
}

void
main()
{
    vec3 N = (gl_FrontFacing ? inpt.v.normal : -inpt.v.normal);
#ifdef USE_PTEX_DISPLACEMENT
    N = perturbNormalFromDisplacement(inpt.v.position.xyz,
                                      N,
                                      inpt.v.patchCoord);
#endif

    vec4 Cf = vec4(1.0);
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
    Cf = edgeColor(Cf, inpt.barycentric);
#endif

#ifdef USE_PTEX_COLOR
    Cf = Cf * (vec4(1) - vec4(PTexLookup(inpt.v.patchCoord,
                                    textureImage_Data,
                                    textureImage_Packing,
                                         textureImage_Pages).x));
#endif

    Cf = lighting(Cf, inpt.v.position.xyz, N);

    outColor = Cf;
}

#endif
