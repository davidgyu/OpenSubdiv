#line 0 "examples/mtlEvalLimit/mtlEvalLimit.metal"

//
//   Copyright 2017 Pixar
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

#include <metal_stdlib>

using namespace metal;

struct PerFrameConstants {
   float4x4 ModelViewMatrix;
   float4x4 ProjectionMatrix;
   float4x4 ModelViewProjectionMatrix;
   float4x4 ModelViewInverseMatrix;
};

struct FragmentData {
    float4 clipPosition [[position]];
    float pointSize [[point_size]];
    float4 color;
};

vertex FragmentData vertex_main(
    uint vertex_id [[vertex_id]],
    constant PerFrameConstants &frameConsts [[buffer(FRAME_CONST_BUFFER_INDEX)]],
    constant uint &drawMode [[buffer(DRAWMODE_BUFFER_INDEX)]],
    constant packed_float3 *vertexBuf [[buffer(VERTEX_BUFFER_INDEX)]],
    constant packed_float3 *deriv1Buf [[buffer(DERIV1_BUFFER_INDEX)]],
    constant packed_float3 *deriv2Buf [[buffer(DERIV2_BUFFER_INDEX)]],
    constant float *patchCoordBuf [[buffer(PATCHCOORD_BUFFER_INDEX)]],
    constant packed_float2 *fvarDataBuf [[buffer(FVARDATA_BUFFER_INDEX)]])
{
    float3 position(vertexBuf[vertex_id*2+0]);
    float3 color(vertexBuf[vertex_id*2+1]);
    float3 du(deriv1Buf[vertex_id*2+0]);
    float3 dv(deriv1Buf[vertex_id*2+1]);
    float3 duu(deriv2Buf[vertex_id*3+0]);
    float3 duv(deriv2Buf[vertex_id*3+1]);
    float3 dvv(deriv2Buf[vertex_id*3+2]);
    float2 patchUV(patchCoordBuf[vertex_id*5+3], patchCoordBuf[vertex_id*5+4]); // s,t
    float2 fvarData(fvarDataBuf[vertex_id]);

    FragmentData out;
    out.clipPosition = frameConsts.ProjectionMatrix * frameConsts.ModelViewMatrix * float4(position, 1);
    out.pointSize = 2.0f;

    float3 normal = (frameConsts.ModelViewMatrix * float4(normalize(cross(du, dv)), 0)).xyz;

    if (drawMode == 0) { // UV
        out.color = float4(patchUV.x, patchUV.y, 0, 1);
    } else if (drawMode == 2) {
        out.color = float4(normal*0.5+float3(0.5), 1);
    } else if (drawMode == 3) {
        out.color = float4(float3(1)*dot(normal, float3(0,0,1)), 1);
    } else if (drawMode == 4) {  // face varying
        // generating a checkerboard pattern
        int checker = int(floor(20*fvarData.r)+floor(20*fvarData.g))&1;
        out.color = float4(fvarData.rg*checker, 1-checker, 1);
    } else if (drawMode == 5) {  // mean curvature
        float3 N = normalize(cross(du, dv));
        float E = dot(du, du);
        float F = dot(du, dv);
        float G = dot(dv, dv);
        float e = dot(N, duu);
        float f = dot(N, duv);
        float g = dot(N, dvv);
        float H = 0.5 * abs(0.5 * (E*g - 2*F*f - G*e) / (E*G - F*F));
        out.color = float4(H, H, H, 1.0);
    } else { // varying
        out.color = float4(color, 1);
    }
    return out;
}

fragment float4 fragment_main(FragmentData in [[stage_in]]) {
    return in.color;
}
