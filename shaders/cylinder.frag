#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragWorldPos;

layout(location = 0) out vec4 outColor;

layout(std140, set = 0, binding = 0) uniform SceneUniforms {
    mat4 view_projection;
    vec3 camera_position;
    vec3 directional_light_direction;
    vec3 directional_light_color;
    vec3 ambient_light;
    float _pad3;
    uint point_light_count;
    uint spot_light_count;
} scene_uniforms;

layout(std140, set = 1, binding = 0) uniform ModelUniform {
    mat4 model;
    vec3 albedo;
    vec3 specular;
    float alpha;
} model_uniforms;

struct PointLight {
    vec3 position;
    vec3 color;
};

// Spotlight с двумя углами (inner/outer) — по заданию лабы
struct Spotlight {
    vec3 position;
    vec3 color;
    vec3 direction;
    float inner_angle;   // градусы
    float outer_angle;   // градусы
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight point_lights[];
};

layout(std430, set = 0, binding = 2) readonly buffer Spotlights {
    Spotlight spot_lights[];
};

void main() {
    vec3 N = normalize(fragNormal);
    vec3 V = normalize(scene_uniforms.camera_position - fragWorldPos);

    vec3 ambient = scene_uniforms.ambient_light * model_uniforms.albedo;

    vec3 diffuseSum  = vec3(0.0);
    vec3 specularSum = vec3(0.0);

    // ---------- Directional ----------
    {
        vec3 L = normalize(-scene_uniforms.directional_light_direction);
        float NdotL = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float spec = (NdotL > 0.0) ? pow(NdotH, max(model_uniforms.alpha, 1.0)) : 0.0;

        diffuseSum  += scene_uniforms.directional_light_color * NdotL * model_uniforms.albedo;
        specularSum += scene_uniforms.directional_light_color * spec * model_uniforms.specular;
    }

    // ---------- Point lights ----------
    for (uint i = 0u; i < scene_uniforms.point_light_count; ++i) {
        PointLight pl = point_lights[i];
        vec3 delta = pl.position - fragWorldPos;
        float dist = length(delta);
        vec3 L = delta / max(dist, 1e-4);

        float attenuation = 1.0 / max(dist * dist, 1e-4);

        float NdotL = max(dot(N, L), 0.0);
        vec3 H = normalize(L + V);
        float NdotH = max(dot(N, H), 0.0);
        float spec = (NdotL > 0.0) ? pow(NdotH, max(model_uniforms.alpha, 1.0)) : 0.0;

        diffuseSum  += attenuation * pl.color * NdotL * model_uniforms.albedo;
        specularSum += attenuation * pl.color * spec * model_uniforms.specular;
    }

    // ---------- Spot lights ----------
    for (uint i = 0u; i < scene_uniforms.spot_light_count; ++i) {
    Spotlight sl = spot_lights[i];

    // L: от фрагмента поверхности к источнику света
    vec3 delta = sl.position - fragWorldPos;
    float dist = length(delta);
    vec3 L = delta / max(dist, 1e-4);

    // Направление от источника к фрагменту
    vec3 lightDir = -L;

    // sl.direction указывает туда, куда светит прожектор
    float cosTheta = dot(
        normalize(sl.direction),
        lightDir
    );

    float innerCos = cos(radians(sl.inner_angle));
    float outerCos = cos(radians(sl.outer_angle));

    float spotFactor = clamp(
        (cosTheta - outerCos) /
        max(innerCos - outerCos, 1e-5),
        0.0,
        1.0
    );

    float attenuation =
        spotFactor / max(dist * dist, 1e-4);

    float NdotL = max(dot(N, L), 0.0);

    vec3 H = normalize(L + V);
    float NdotH = max(dot(N, H), 0.0);

    float spec = (NdotL > 0.0)
        ? pow(NdotH, max(model_uniforms.alpha, 1.0))
        : 0.0;

    diffuseSum +=
        attenuation * sl.color * NdotL * model_uniforms.albedo;

    specularSum +=
        attenuation * sl.color * spec * model_uniforms.specular;
    }

    vec3 finalColor = ambient + diffuseSum + specularSum;
    outColor = vec4(finalColor, 1.0);
}