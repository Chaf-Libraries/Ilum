struct DirectionalLight
{
    vec3 color;
    float intensity;
    vec3 direction;
};

struct PointLight
{
    vec3 color;
    float intensity;
    vec3 position;
    float constant;
    float linear;
    float quadratic;
};

struct SpotLight
{
    vec3 color;
    float intensity;
    vec3 position;
    float cut_off;
    vec3 direction;
    float outer_cut_off;
};