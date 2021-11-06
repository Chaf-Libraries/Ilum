#version 450

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec2 inUV;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec3 inTangent;
layout(location = 4) in vec3 inBiTangent;

layout(location = 0) out vec4 outColor;

layout (set = 0, binding = 0) uniform MainCamera
{
    mat4 view_projection;
    vec3 position;
}main_camera;

void main()
{
    float ambient = 0.1;

    vec3 norm = normalize(inNormal);
    vec3 light_dir = normalize(vec3(-1,-1,-1));

    vec3 diffuse = vec3(max(dot(norm, light_dir), 0.0));

    vec3 view_dir = normalize(main_camera.position-inPos);
    vec3 reflect_dir = reflect(-light_dir, norm);
    vec3 spec = vec3(pow(max(dot(view_dir, reflect_dir), 0.0), 32));

    outColor = vec4(ambient + diffuse + spec, 1.0);
}