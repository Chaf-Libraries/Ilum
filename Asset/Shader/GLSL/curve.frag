#version 450

layout(location = 0) in vec4 inColor;
layout(location = 1) flat in uint inInstanceID;
layout(location = 2) flat in uint inEntityID;

layout(location = 0) out vec4 Albedo;
layout(location = 1) out vec4 Instance_Vis;
layout(location = 2) out uint Entity_ID;

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main()
{
    Albedo = inColor;
    Instance_Vis = vec4(rand(vec2(inInstanceID)), rand(vec2(inInstanceID + 1)), rand(vec2(inInstanceID + 2)), 1);;
    Entity_ID = inEntityID;
}