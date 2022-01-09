#version 450

layout(binding = 0) uniform sampler2D Light;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

layout(push_constant) uniform PushBlock{
    float threshold;
	float scale;
	float strength;
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
        outColor=texture(Light, inUV);
    }
    else
    {
	    const float weights[] = float[](0.0024499299678342,
									0.0043538453346397,
									0.0073599963704157,
									0.0118349786570722,
									0.0181026699707781,
									0.0263392293891488,
									0.0364543006660986,
									0.0479932050577658,
									0.0601029809166942,
									0.0715974486241365,
									0.0811305381519717,
									0.0874493212267511,
									0.0896631113333857,
									0.0874493212267511,
									0.0811305381519717,
									0.0715974486241365,
									0.0601029809166942,
									0.0479932050577658,
									0.0364543006660986,
									0.0263392293891488,
									0.0181026699707781,
									0.0118349786570722,
									0.0073599963704157,
									0.0043538453346397,
									0.0024499299678342);

        vec2 tex_offset = 1.0 / textureSize(Light, 0) * push_data.scale;

        vec3 color = vec3(0.0);

        vec2 start = inUV - vec2(tex_offset.x * 2, tex_offset.y * 2);

        for (int i = 0; i < 5; i++)
        {
            for(int j = 0; j < 5; j++)
            {
                color += preprocess(texture(Light, start + vec2(tex_offset.x * i, tex_offset.y * j)).rgb) * weights[i * 5 + j] * push_data.strength;
            }
        }

        outColor = vec4(color + texture(Light, inUV).rgb, 1.0);
    }
}