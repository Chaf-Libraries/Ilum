#version 450

layout(location = 0) in vec4 inColor;
layout(location = 1) flat in uint inEntityID;

layout(location = 0) out vec4 CurveView;
layout(location = 1) out uint Entity_ID;

float rand(vec2 co){
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main()
{
    CurveView = inColor;
    Entity_ID = inEntityID;
}