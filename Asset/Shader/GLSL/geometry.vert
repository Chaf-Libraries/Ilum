#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out vec3 outPos;
layout(location = 1) out vec2 outUV;
layout(location = 2) out vec3 outNormal;
layout(location = 3) out vec3 outTangent;
layout(location = 4) out vec3 outBiTangent;
layout(location = 5) out vec4 outScreenPos;

void main() {
    // flip Y
    gl_Position = vec4(inPos.xy, (inPos.z+1.0)/10, 1.0);
    outScreenPos = gl_Position;

    outPos = vec3(inPos);
    outUV = inUV;
    outNormal = normalize(inNormal);
    outTangent = normalize(inTangent);
    outBiTangent = normalize(inBiTangent);
}