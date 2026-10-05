#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform ColorUBO {
    vec4 userColor;
} colorUBO;

void main() {
    outColor = colorUBO.userColor * vec4(fragColor, 1.0);
}