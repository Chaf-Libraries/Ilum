#version 450

layout(binding = 0) uniform sampler2D Light;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushBlock{
	float scale;
	float strength;
	uint horizental;
}push_data;

void main()
{
    float weights[5];
    weights[0] = 0.227027;
    weights[1] = 0.1945946;
    weights[2] = 0.1216216;
    weights[3] = 0.054054;
    weights[4] = 0.016216;

    vec2 tex_offset = 1.0 / textureSize(Light, 0) * push_data.scale; // gets size of single texel
    vec3 result = texture(Light, inUV).rgb * weights[0]; // current fragment's contribution
    for(int i = 1; i < 5; ++i)
    {
        if (push_data.horizental == 1)
        {
            // H
            result += texture(Light, inUV + vec2(tex_offset.x * i, 0.0)).rgb * weights[i] * push_data.strength;
            result += texture(Light, inUV - vec2(tex_offset.x * i, 0.0)).rgb * weights[i] * push_data.strength;
        }
        else
        {
            // V
            result += texture(Light, inUV + vec2(0.0, tex_offset.y * i)).rgb * weights[i] * push_data.strength;
            result += texture(Light, inUV - vec2(0.0, tex_offset.y * i)).rgb * weights[i] * push_data.strength;
        }
    }
    outColor = vec4(result, 1.0);
}