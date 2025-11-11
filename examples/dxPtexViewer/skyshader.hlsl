//
//   Copyright 2013 Nvidia
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

struct VS_InputVertex {
    float3 position : POSITION0;
    float2 texCoord : TEXCOORD0;
};

struct VS_OutputVertex {
    float4 position : SV_POSITION0;
    float2 texCoord : TEXCOORD0;
};

cbuffer Transform : register( b0 ) {
    float4x4 ModelViewMatrix;
};

//--------------------------------------------------------------
// sky vertex shader
//--------------------------------------------------------------


void vs_main(in VS_InputVertex input,
             out VS_OutputVertex output) {

    output.position = mul(ModelViewMatrix, float4(input.position,1));
    output.texCoord = input.texCoord;
}

//--------------------------------------------------------------
// sky pixel shader
//--------------------------------------------------------------

struct PS_InputVertex {
    float4 position : SV_POSITION0;
    float2 texCoord : TEXCOORD0;
};

Texture2D tx : register(t0);

SamplerState sm : register(s0);

float4
gamma(float4 value, float g) {
    return float4(pow(value.xyz, float3(g,g,g)), 1);
}


float4
ps_main(in PS_InputVertex input) : SV_Target {

    float4 tex = tx.Sample(sm, input.texCoord.xy);

    //float4 outColor = gamma(tex,0.4545);
    float4 outColor = tex;

    return outColor;
}

