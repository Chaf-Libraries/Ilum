#version 450

layout(binding = 0) uniform sampler2D Albedo;
layout(binding = 1) uniform sampler2D Normal;
layout(binding = 2) uniform sampler2D Position;
layout(binding = 3) uniform sampler2D Depth;
layout(binding = 4) uniform sampler2D Metallic;
layout(binding = 5) uniform sampler2D Roughness;

layout (location = 0) in vec2 inUV;

layout (location = 0) out vec4 outColor;

layout(push_constant) uniform Light{
    float intensity;
    vec3 color;
    vec3 direction;
    vec3 view_pos;
}light;

void main()
{
    vec4 albedo = texture(Albedo, inUV);
    vec3 normal = texture(Normal, inUV).rgb;
    vec3 frag_pos = texture(Position, inUV).rgb;
    float metallic = texture(Metallic, inUV).r;
    float roughness = texture(Roughness, inUV).r;

    vec3 L = normalize(light.direction);
    vec3 V = normalize(light.view_pos - frag_pos);
    
    // Diffuse
    vec3 N = normalize(normal);
    float NoL = max(0.0, dot(N,L));
    vec3 diff = light.color * albedo.rgb * NoL * light.intensity;
    
    outColor=vec4(diff, 1.0);
}