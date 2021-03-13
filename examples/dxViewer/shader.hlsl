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

struct OutputPointVertex {
    float4 positionOut : SV_Position;
};

cbuffer Transform : register( b0 ) {
    float4x4 ModelViewMatrix;
    float4x4 ProjectionMatrix;
    float4x4 ModelViewProjectionMatrix;
    float4x4 ModelViewInverseMatrix;
};

cbuffer Tessellation : register( b1 ) {
    float TessLevel;
    int GregoryQuadOffsetBase;
    int PrimitiveIdBase;
};

float4x4 OsdModelViewMatrix()
{
    return ModelViewMatrix;
}
float4x4 OsdProjectionMatrix()
{
    return ProjectionMatrix;
}
float4x4 OsdModelViewProjectionMatrix()
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

// ---------------------------------------------------------------------------
//  Vertex Shader
// ---------------------------------------------------------------------------

void vs_main( in InputVertex input,
              out OutputVertex output )
{
    output.positionOut = mul(ModelViewProjectionMatrix, input.position);
    output.position = mul(ModelViewMatrix, input.position);
    output.normal = mul(ModelViewMatrix,float4(input.normal, 0)).xyz;
}

// ---------------------------------------------------------------------------
//  Geometry Shader
// ---------------------------------------------------------------------------

struct GS_OUT
{
    OutputVertex v;
    uint primitiveID : SV_PrimitiveID;
    noperspective float3 barycentric : BARYCENTRIC;
};

GS_OUT
outputVertex(int index, OutputVertex input, float3 normal, uint primitiveID)
{
    GS_OUT gsout;
    gsout.v = input;
    gsout.v.normal = normal;
    gsout.primitiveID = primitiveID;

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    float3 coords[3] = {float3(0,0,1),float3(1,0,0),float3(0,1,0)};
    gsout.barycentric = coords[index];
#endif
#ifdef PRIM_QUAD
    float3 coords[4] = {float3(0,0,1),float3(1,0,0),float3(0,0,1),float3(0,1,0)};
    gsout.barycentric = coords[index];
#endif
#else
    gsout.barycentric = float3(0,0,0);
#endif

    return gsout;
}

[maxvertexcount(6)]
void gs_quad( lineadj OutputVertex input[4],
              uint primitiveID : SV_PrimitiveID,
              inout TriangleStream<GS_OUT> triStream )
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[3].position - input[1].position).xyz;
    float3 C = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    triStream.Append(outputVertex(0, input[0], n0, primitiveID));
    triStream.Append(outputVertex(1, input[1], n0, primitiveID));
    triStream.Append(outputVertex(3, input[3], n0, primitiveID));
    triStream.RestartStrip();
    triStream.Append(outputVertex(3, input[3], n0, primitiveID));
    triStream.Append(outputVertex(1, input[1], n0, primitiveID));
    triStream.Append(outputVertex(2, input[2], n0, primitiveID));
    triStream.RestartStrip();
}

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_QUAD
[maxvertexcount(6)]
void gs_quad_wire( lineadj OutputVertex input[4],
              uint primitiveID : SV_PrimitiveID,
              inout TriangleStream<GS_OUT> triStream )
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[3].position - input[1].position).xyz;
    float3 C = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    triStream.Append(outputVertex(0, input[0], n0, primitiveID));
    triStream.Append(outputVertex(1, input[1], n0, primitiveID));
    triStream.Append(outputVertex(3, input[3], n0, primitiveID));
    triStream.RestartStrip();
    triStream.Append(outputVertex(3, input[3], n0, primitiveID));
    triStream.Append(outputVertex(1, input[1], n0, primitiveID));
    triStream.Append(outputVertex(2, input[2], n0, primitiveID));
    triStream.RestartStrip();
}
#endif
#endif

[maxvertexcount(3)]
void gs_triangle( triangle OutputVertex input[3],
                  uint primitiveID : SV_PrimitiveID,
                  inout TriangleStream<GS_OUT> triStream )
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    triStream.Append(outputVertex(0, input[0], n0, primitiveID));
    triStream.Append(outputVertex(1, input[1], n0, primitiveID));
    triStream.Append(outputVertex(2, input[2], n0, primitiveID));
}

[maxvertexcount(3)]
void gs_triangle_smooth( triangle OutputVertex input[3],
                         uint primitiveID : SV_PrimitiveID,
                         inout TriangleStream<GS_OUT> triStream )
{
    triStream.Append(outputVertex(0, input[0], input[0].normal, primitiveID));
    triStream.Append(outputVertex(1, input[1], input[1].normal, primitiveID));
    triStream.Append(outputVertex(2, input[2], input[2].normal, primitiveID));
}

#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
[maxvertexcount(3)]
void gs_triangle_wire( triangle OutputVertex input[3],
                       uint primitiveID : SV_PrimitiveID,
                       inout TriangleStream<GS_OUT> triStream )
{
    float3 A = (input[0].position - input[1].position).xyz;
    float3 B = (input[2].position - input[1].position).xyz;

    float3 n0 = normalize(cross(B, A));

    triStream.Append(outputVertex(0, input[0], n0, primitiveID));
    triStream.Append(outputVertex(1, input[1], n0, primitiveID));
    triStream.Append(outputVertex(2, input[2], n0, primitiveID));
}

[maxvertexcount(3)]
void gs_triangle_smooth_wire( triangle OutputVertex input[3],
                              uint primitiveID : SV_PrimitiveID,
                              inout TriangleStream<GS_OUT> triStream )
{
    triStream.Append(outputVertex(0, input[0], input[0].normal, primitiveID));
    triStream.Append(outputVertex(1, input[1], input[1].normal, primitiveID));
    triStream.Append(outputVertex(2, input[2], input[2].normal, primitiveID));
}
#endif
#endif

[maxvertexcount(1)]
void gs_point( point OutputVertex input[1],
               inout PointStream<OutputPointVertex> pointStream )
{
    OutputPointVertex v0;
    v0.positionOut = input[0].positionOut;

    pointStream.Append(v0);
}

// ---------------------------------------------------------------------------
//  Lighting
// ---------------------------------------------------------------------------

#define NUM_LIGHTS 2

struct LightSource {
    float4 position;
    float4 ambient;
    float4 diffuse;
    float4 specular;
};

cbuffer Lighting : register( b2 ) {
    LightSource lightSource[NUM_LIGHTS];
};


cbuffer Material : register( b3 ){
    float4 materialColor;
}

float4
lighting(float4 diffuse, float3 Peye, float3 Neye)
{
    float4 color = float4(0.0, 0.0, 0.0, 0.0);
    //float4 material = float4(0.4, 0.4, 0.8, 1);
    //float4 material = float4(0.13, 0.13, 0.61, 1); // sRGB (gamma 2.2)

    for (int i = 0; i < NUM_LIGHTS; ++i) {

        float4 Plight = lightSource[i].position;

        float3 l = (Plight.w == 0.0)
                    ? normalize(Plight.xyz) : normalize(Plight.xyz - Peye);

        float3 n = normalize(Neye);
        float3 h = normalize(l + float3(0,0,1));    // directional viewer

        float d = max(0.0, dot(n, l));
        float s = pow(max(0.0, dot(n, h)), 500.0f);

        color += lightSource[i].ambient
            + d * lightSource[i].diffuse * diffuse
            + s * lightSource[i].specular;
    }


    color.a = 1.0;
    return color;
}

// ---------------------------------------------------------------------------
//  Pixel Shader
// ---------------------------------------------------------------------------

float4
edgeColor(float4 Cfill, float3 barycentric)
{
#if defined(GEOMETRY_OUT_WIRE) || defined(GEOMETRY_OUT_LINE)
#ifdef PRIM_TRI
    float3 dist = max(float3(0,0,0),barycentric / fwidth(barycentric));
#endif
#ifdef PRIM_QUAD
    float3 dist = max(float3(0,0,0),barycentric / fwidth(float3(barycentric.xy,1)));
#endif

    float d = min(dist.x, min(dist.y, dist.z));
    float p = exp2(-2 * d * d);

#if defined(GEOMETRY_OUT_WIRE)
    if (p < 0.25) discard;
#endif

    float v = 0.8;
    float4 Cedge = float4(Cfill.r*v, Cfill.g*v, Cfill.b*v, 1);

    Cfill.rgb = lerp(Cfill.rgb, Cedge.rgb, p);
#endif
    return Cfill;
}

float4
getAdaptivePatchColor(int3 patchParam, float2 vSegments)
{
    const float4 patchColors[7*6] = {
        float4(1.0f,  1.0f,  1.0f,  1.0f),   // regular
        float4(0.0f,  1.0f,  1.0f,  1.0f),   // regular pattern 0
        float4(0.0f,  0.5f,  1.0f,  1.0f),   // regular pattern 1
        float4(0.0f,  0.5f,  0.5f,  1.0f),   // regular pattern 2
        float4(0.5f,  0.0f,  1.0f,  1.0f),   // regular pattern 3
        float4(1.0f,  0.5f,  1.0f,  1.0f),   // regular pattern 4

        float4(1.0f,  0.5f,  0.5f,  1.0f),   // single crease
        float4(1.0f,  0.70f,  0.6f,  1.0f),  // single crease pattern 0
        float4(1.0f,  0.65f,  0.6f,  1.0f),  // single crease pattern 1
        float4(1.0f,  0.60f,  0.6f,  1.0f),  // single crease pattern 2
        float4(1.0f,  0.55f,  0.6f,  1.0f),  // single crease pattern 3
        float4(1.0f,  0.50f,  0.6f,  1.0f),  // single crease pattern 4

        float4(0.8f,  0.0f,  0.0f,  1.0f),   // boundary
        float4(0.0f,  0.0f,  0.75f, 1.0f),   // boundary pattern 0
        float4(0.0f,  0.2f,  0.75f, 1.0f),   // boundary pattern 1
        float4(0.0f,  0.4f,  0.75f, 1.0f),   // boundary pattern 2
        float4(0.0f,  0.6f,  0.75f, 1.0f),   // boundary pattern 3
        float4(0.0f,  0.8f,  0.75f, 1.0f),   // boundary pattern 4

        float4(0.0f,  1.0f,  0.0f,  1.0f),   // corner
        float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 0
        float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 1
        float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 2
        float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 3
        float4(0.5f,  1.0f,  0.5f,  1.0f),   // corner pattern 4

        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory
        float4(1.0f,  1.0f,  0.0f,  1.0f),   // gregory

        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary
        float4(1.0f,  0.5f,  0.0f,  1.0f),   // gregory boundary

        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f),   // gregory basis
        float4(1.0f,  0.7f,  0.3f,  1.0f)    // gregory basis
    };

    int patchType = 0;

    int edgeCount = countbits(OsdGetPatchBoundaryMask(patchParam));
    if (edgeCount == 1) {
        patchType = 2; // BOUNDARY
    }
    if (edgeCount > 1) {
        patchType = 3; // CORNER (not correct for patches that are not isolated)
    }

#if defined OSD_PATCH_ENABLE_SINGLE_CREASE
    if (vSegments.y > 0) {
        patchType = 1;
    }
#elif defined OSD_PATCH_GREGORY
    patchType = 4;
#elif defined OSD_PATCH_GREGORY_BOUNDARY
    patchType = 5;
#elif defined OSD_PATCH_GREGORY_BASIS
    patchType = 6;
#elif defined OSD_PATCH_GREGORY_TRIANGLE
    patchType = 6;
#endif

    int pattern = countbits(OsdGetPatchTransitionMask(patchParam));

    return patchColors[6*patchType + pattern];
}

float4
getAdaptiveDepthColor(int3 patchParam)
{
    //  Represent depth with repeating cycle of four colors:
    const float4 depthColors[4] = {
        float4(0.0f,  0.5f,  0.5f,  1.0f),
        float4(1.0f,  1.0f,  1.0f,  1.0f),
        float4(0.0f,  1.0f,  1.0f,  1.0f),
        float4(0.5f,  1.0f,  0.5f,  1.0f)
    };
    return depthColors[OsdGetPatchRefinementLevel(patchParam) & 3];
}


// ---------------------------------------------------------------------------
//  Pixel Shader
// ---------------------------------------------------------------------------

void
ps_main( in OutputVertex input,
         uint primitiveID : SV_PrimitiveID,
         float3 barycentric : BARYCENTRIC,
         bool isFrontFacing : SV_IsFrontFace,
         out float4 colorOut : SV_Target )
{
    float2 vSegments = float2(0,0);
#ifdef OSD_PATCH_ENABLE_SINGLE_CREASE
    vSegments = input.vSegments;
#endif


#if defined(SHADING_PATCH_TYPE)
    float4 color = getAdaptivePatchColor(
        OsdGetPatchParam(OsdGetPatchIndex(primitiveID)), vSegments);
#elif defined(SHADING_PATCH_DEPTH)
    float4 color = getAdaptiveDepthColor(
        OsdGetPatchParam(OsdGetPatchIndex(primitiveID)));
#elif defined(SHADING_PATCH_COORD)
    float4 color = float4(input.patchCoord.x, input.patchCoord.y, 0, 1);
#elif defined(SHADING_MATERIAL)
    float4 color = float4(0.4, 0.4, 0.8, 1.0);
#else
    float4 color = float4(1, 1, 1, 1);
#endif

    float3 N = (isFrontFacing ? input.normal : -input.normal);
    float4 Cf = lighting(color, input.position.xyz, N);

#if defined(SHADING_NORMAL)
    Cf.rgb = N;
#endif

    colorOut = edgeColor(Cf, barycentric);
}

void
ps_main_point( in OutputPointVertex input,
               out float4 colorOut : SV_Target )
{
    colorOut = float4(1, 1, 1, 1);
}
