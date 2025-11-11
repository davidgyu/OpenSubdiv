//
//   Copyright 2013 Pixar
//
//   Licensed under the terms set forth in the LICENSE.txt file available at
//   https://opensubdiv.org/license.
//

//--------------------------------------------------------------
// sky vertex shader
//--------------------------------------------------------------
#ifdef SKY_VERTEX_SHADER

layout (location=0) in vec3 position;
layout (location=1) in vec2 texCoord;

uniform mat4 ModelViewProjectionMatrix;

out vec2 outTexCoord;

void
main()
{
    gl_Position = ModelViewProjectionMatrix * vec4(position, 1);
    outTexCoord = texCoord.xy;
}

#endif

//--------------------------------------------------------------
// sky fragment shader
//--------------------------------------------------------------
#ifdef SKY_FRAGMENT_SHADER

uniform sampler2D environmentMap;

in vec2 outTexCoord;
out vec4 outColor;

vec4 getEnvironmentHDR(sampler2D sampler, vec2 uv)
{
    vec4 tex = texture(sampler, uv);
    tex = vec4(pow(tex.xyz, vec3(0.4545)), 1);
    return tex;
}

void
main()
{
    outColor = getEnvironmentHDR(environmentMap, outTexCoord);
}

#endif
