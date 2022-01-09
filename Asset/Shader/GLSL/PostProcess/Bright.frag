#version 450

layout(binding = 0) uniform sampler2D Light;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushBlock{
    float threshold;
    uint enable;
}push_data;

vec3 preprocess(vec3 color)
{
    vec3 result;
    float l = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
	result = (l > push_data.threshold) ? color.rgb : vec3(0.0);
    return result;
}

void main()
{
    if(push_data.enable == 0)
    {
        outColor=vec4(vec3(0.0),1.0);
    }
    else
    {
        outColor = vec4(preprocess(texture(Light, inUV).rgb), 1.0);
    }
}