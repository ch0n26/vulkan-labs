#include "application.hpp"

#include <imgui.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace application {

    namespace {

        constexpr int MAX_POINT_LIGHTS = 16;
        constexpr int MAX_SPOT_LIGHTS = 16;

        // ============================================================
        //  Структуры (совпадают с GLSL)
        // ============================================================

        struct Vertex {
            glm::vec3 pos;
            glm::vec3 color;
            glm::vec3 normal;

            static VkVertexInputBindingDescription bindingDescription() {
                return { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
            }

            static std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions() {
                return { {
                    { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                    { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                    { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                } };
            }
        };

        struct SceneUniforms {
            glm::mat4 view_projection;
            glm::vec3 camera_position;
            float     _pad0 = 0.0f;
            glm::vec3 directional_light_direction;
            float     _pad1 = 0.0f;
            glm::vec3 directional_light_color;
            float     _pad2 = 0.0f;
            glm::vec3 ambient_light;
            float     _pad3 = 0.0f;
            uint32_t  point_light_count = 0;
            uint32_t  spot_light_count = 0;
            uint32_t  _pad4[2] = { 0, 0 };
        };

        struct ModelUniform {
            glm::mat4 model;
            glm::vec3 albedo;
            float     _pad0 = 0.0f;
            glm::vec3 specular;
            float     alpha = 32.0f;
        };

        struct alignas(16) PointLight {
            glm::vec3 position;
            float     _pad0 = 0;
            glm::vec3 color;
            float     _pad1 = 0;
        };

        // Spotlight с двумя углами (inner/outer) — по заданию лабы
        struct alignas(16) Spotlight {
            glm::vec3 position;
            float     _pad0 = 0;
            glm::vec3 color;
            float     _pad1 = 0;
            glm::vec3 direction;
            float     inner_angle = 15.0f;   // градусы
            float     outer_angle = 25.0f;   // градусы, должно быть > inner
            float     _pad2[2] = { 0, 0 };
        };

        static_assert(sizeof(PointLight) == 32, "PointLight must be 32 bytes");
        static_assert(sizeof(Spotlight) == 64, "Spotlight must be 64 bytes");

        // ============================================================
        //  Vulkan-объекты
        // ============================================================
        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline       vk_pipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout vk_scene_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout vk_model_set_layout = VK_NULL_HANDLE;

        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet  vk_scene_set = VK_NULL_HANDLE;
        VkDescriptorSet  vk_model_sets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

        VkBuffer      vk_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vma_vertex_buffer = VK_NULL_HANDLE;
        uint32_t      vertex_count = 0;

        VkBuffer      vk_index_buffer = VK_NULL_HANDLE;
        VmaAllocation vma_index_buffer = VK_NULL_HANDLE;
        uint32_t      index_count = 0;

        VkBuffer      vk_scene_ubo = VK_NULL_HANDLE;
        VmaAllocation vma_scene_ubo = VK_NULL_HANDLE;
        void* scene_ubo_mapped = nullptr;

        VkBuffer      vk_point_light_ssbo = VK_NULL_HANDLE;
        VmaAllocation vma_point_light_ssbo = VK_NULL_HANDLE;
        void* point_light_ssbo_mapped = nullptr;

        VkBuffer      vk_spot_light_ssbo = VK_NULL_HANDLE;
        VmaAllocation vma_spot_light_ssbo = VK_NULL_HANDLE;
        void* spot_light_ssbo_mapped = nullptr;

        VkBuffer      vk_model_ubos[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        VmaAllocation vma_model_ubos[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
        void* model_ubo_mapped[2] = { nullptr, nullptr };

        // ============================================================
        //  Состояние сцены
        // ============================================================
        struct ObjectState {
            glm::vec3 position{ 0.0f };
            glm::vec3 rotation{ 0.0f };
            glm::vec3 scale{ 1.0f };
            glm::vec3 albedo{ 1.0f };
            glm::vec3 specular{ 1.0f };
            float     alpha = 32.0f;
        };

        ObjectState obj1;
        ObjectState obj2;

        float camera_yaw = 0.0f;
        float camera_pitch = 20.0f;
        float camera_distance = 6.0f;
        float camera_fov = 45.0f;

        glm::vec3 ambient_light{ 0.1f };
        glm::vec3 directional_direction{ -0.3f, -1.0f, -0.5f };
        glm::vec3 directional_color{ 1.5f };

        std::vector<PointLight> point_lights;
        std::vector<Spotlight>  spot_lights;

        // ============================================================
        //  Вспомогательное
        // ============================================================
        std::vector<char> readFile(const std::string& path) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << path << '\n';
                return {};
            }
            size_t size = static_cast<size_t>(file.tellg());
            std::vector<char> buffer(size);
            file.seekg(0);
            file.read(buffer.data(), size);
            return buffer;
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) {
            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = code.size();
            info.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
                std::cerr << "Failed to create shader module\n";
                return VK_NULL_HANDLE;
            }
            return module;
        }

        std::vector<Vertex> generateCylinder(float radius, float height, int segments) {
            std::vector<Vertex> vertices;
            const float halfH = height / 2.0f;
            const float PI = 3.14159265f;

            auto makeVertex = [&](float x, float y, float z, float nx, float ny, float nz) -> Vertex {
                float t = (y + halfH) / height;
                glm::vec3 color(t, 1.0f - t, 0.5f);
                return { glm::vec3(x, y, z), color, glm::vec3(nx, ny, nz) };
                };

            for (int i = 0; i < segments; ++i) {
                float a0 = 2.0f * PI * i / segments;
                float a1 = 2.0f * PI * (i + 1) / segments;
                float c0 = cosf(a0), s0 = sinf(a0);
                float c1 = cosf(a1), s1 = sinf(a1);

                float x0 = radius * c0, z0 = radius * s0;
                float x1 = radius * c1, z1 = radius * s1;

                vertices.push_back(makeVertex(x0, -halfH, z0, c0, 0.0f, s0));
                vertices.push_back(makeVertex(x1, -halfH, z1, c1, 0.0f, s1));
                vertices.push_back(makeVertex(x1, halfH, z1, c1, 0.0f, s1));

                vertices.push_back(makeVertex(x0, -halfH, z0, c0, 0.0f, s0));
                vertices.push_back(makeVertex(x1, halfH, z1, c1, 0.0f, s1));
                vertices.push_back(makeVertex(x0, halfH, z0, c0, 0.0f, s0));
            }

            for (int i = 0; i < segments; ++i) {
                float a0 = 2.0f * PI * i / segments;
                float a1 = 2.0f * PI * (i + 1) / segments;
                vertices.push_back(makeVertex(0.0f, -halfH, 0.0f, 0.0f, -1.0f, 0.0f));
                vertices.push_back(makeVertex(radius * cosf(a1), -halfH, radius * sinf(a1), 0.0f, -1.0f, 0.0f));
                vertices.push_back(makeVertex(radius * cosf(a0), -halfH, radius * sinf(a0), 0.0f, -1.0f, 0.0f));
            }

            for (int i = 0; i < segments; ++i) {
                float a0 = 2.0f * PI * i / segments;
                float a1 = 2.0f * PI * (i + 1) / segments;
                vertices.push_back(makeVertex(0.0f, halfH, 0.0f, 0.0f, 1.0f, 0.0f));
                vertices.push_back(makeVertex(radius * cosf(a0), halfH, radius * sinf(a0), 0.0f, 1.0f, 0.0f));
                vertices.push_back(makeVertex(radius * cosf(a1), halfH, radius * sinf(a1), 0.0f, 1.0f, 0.0f));
            }

            return vertices;
        }

        std::vector<uint32_t> generateIndices(uint32_t count) {
            std::vector<uint32_t> indices(count);
            for (uint32_t i = 0; i < count; ++i) indices[i] = i;
            return indices;
        }

        bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
            VkBuffer& buffer, VmaAllocation& allocation, void** mapped = nullptr) {

            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = size;
            bufferInfo.usage = usage;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
            if (mapped) {
                allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;
            }

            VmaAllocationInfo info{};
            VkResult res = vmaCreateBuffer(graphics::internal::context.allocator,
                &bufferInfo, &allocInfo, &buffer, &allocation, &info);

            if (res != VK_SUCCESS) {
                std::cerr << "vmaCreateBuffer failed: " << res << "\n";
                buffer = VK_NULL_HANDLE;
                allocation = VK_NULL_HANDLE;
                return false;
            }

            if (buffer == VK_NULL_HANDLE) {
                std::cerr << "vmaCreateBuffer returned VK_NULL_HANDLE!\n";
                return false;
            }

            if (mapped) {
                *mapped = info.pMappedData;
                if (*mapped == nullptr) {
                    std::cerr << "WARNING: mapped pointer is null!\n";
                }
            }

            return true;
        }

        void flushBuffer(VmaAllocation alloc) {
            if (alloc == VK_NULL_HANDLE) return;

            VmaAllocationInfo info{};
            vmaGetAllocationInfo(graphics::internal::context.allocator, alloc, &info);

            VkMappedMemoryRange range{};
            range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
            range.memory = info.deviceMemory;
            range.offset = info.offset;
            range.size = VK_WHOLE_SIZE;
            vkFlushMappedMemoryRanges(graphics::internal::context.device, 1, &range);
        }

        // ============================================================
        //  Пайплайн
        // ============================================================
        bool createGraphicsPipeline() {
            VkDevice device = graphics::internal::context.device;

            auto vertCode = readFile("shaders/cylinder.vert.spv");
            auto fragCode = readFile("shaders/cylinder.frag.spv");
            if (vertCode.empty() || fragCode.empty()) {
                std::cerr << "Shader SPIR-V not found.\n";
                return false;
            }

            VkShaderModule vertModule = createShaderModule(vertCode);
            VkShaderModule fragModule = createShaderModule(fragCode);
            if (!vertModule || !fragModule) return false;

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vertModule;
            stages[0].pName = "main";
            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = fragModule;
            stages[1].pName = "main";

            auto binding = Vertex::bindingDescription();
            auto attrs = Vertex::attributeDescriptions();
            VkPipelineVertexInputStateCreateInfo vertexInput{};
            vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertexInput.vertexBindingDescriptionCount = 1;
            vertexInput.pVertexBindingDescriptions = &binding;
            vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attrs.size());
            vertexInput.pVertexAttributeDescriptions = attrs.data();

            VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
            inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewportState{};
            viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewportState.viewportCount = 1;
            viewportState.scissorCount = 1;

            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_BACK_BIT;
            raster.frontFace = VK_FRONT_FACE_CLOCKWISE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depthStencil{};
            depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depthStencil.depthTestEnable = VK_TRUE;
            depthStencil.depthWriteEnable = VK_TRUE;
            depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

            VkPipelineColorBlendAttachmentState colorBlendAttachment{};
            colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            colorBlendAttachment.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo colorBlend{};
            colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            colorBlend.attachmentCount = 1;
            colorBlend.pAttachments = &colorBlendAttachment;

            VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamic{};
            dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic.dynamicStateCount = 2;
            dynamic.pDynamicStates = dynamicStates;

            // Scene set layout (set = 0): UBO + 2 storage buffers
            {
                VkDescriptorSetLayoutBinding b[3]{};
                b[0].binding = 0;
                b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                b[0].descriptorCount = 1;
                b[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

                b[1].binding = 1;
                b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                b[1].descriptorCount = 1;
                b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                b[2].binding = 2;
                b[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                b[2].descriptorCount = 1;
                b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                VkDescriptorSetLayoutCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                info.bindingCount = 3;
                info.pBindings = b;
                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &vk_scene_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create scene set layout\n";
                    return false;
                }
            }

            // Model set layout (set = 1): UBO
            {
                VkDescriptorSetLayoutBinding b{};
                b.binding = 0;
                b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                b.descriptorCount = 1;
                b.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

                VkDescriptorSetLayoutCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                info.bindingCount = 1;
                info.pBindings = &b;
                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &vk_model_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create model set layout\n";
                    return false;
                }
            }

            VkDescriptorSetLayout setLayouts[2] = { vk_scene_set_layout, vk_model_set_layout };
            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 2;
            pipelineLayoutInfo.pSetLayouts = setLayouts;
            if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &vk_pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create pipeline layout\n";
                return false;
            }

            VkGraphicsPipelineCreateInfo pipelineInfo{};
            pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            pipelineInfo.stageCount = 2;
            pipelineInfo.pStages = stages;
            pipelineInfo.pVertexInputState = &vertexInput;
            pipelineInfo.pInputAssemblyState = &inputAssembly;
            pipelineInfo.pViewportState = &viewportState;
            pipelineInfo.pRasterizationState = &raster;
            pipelineInfo.pMultisampleState = &multisample;
            pipelineInfo.pDepthStencilState = &depthStencil;
            pipelineInfo.pColorBlendState = &colorBlend;
            pipelineInfo.pDynamicState = &dynamic;
            pipelineInfo.layout = vk_pipeline_layout;
            pipelineInfo.renderPass = graphics::internal::context.render_pass;
            pipelineInfo.subpass = 0;

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &vk_pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create graphics pipeline\n";
                return false;
            }

            vkDestroyShaderModule(device, vertModule, nullptr);
            vkDestroyShaderModule(device, fragModule, nullptr);
            return true;
        }

        // ============================================================
        //  Дескрипторы
        // ============================================================
        bool createDescriptorSets() {
            VkDevice device = graphics::internal::context.device;

            VkDescriptorPoolSize poolSizes[2]{};
            poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            poolSizes[0].descriptorCount = 1 + 2;
            poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSizes[1].descriptorCount = 2;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 2;
            poolInfo.pPoolSizes = poolSizes;
            poolInfo.maxSets = 3;
            if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &vk_descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor pool\n";
                return false;
            }

            // Scene set
            {
                VkDescriptorSetAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool = vk_descriptor_pool;
                ai.descriptorSetCount = 1;
                ai.pSetLayouts = &vk_scene_set_layout;
                if (vkAllocateDescriptorSets(device, &ai, &vk_scene_set) != VK_SUCCESS) {
                    std::cerr << "Failed to allocate scene set\n";
                    return false;
                }

                VkDescriptorBufferInfo sceneInfo{ vk_scene_ubo, 0, sizeof(SceneUniforms) };
                VkDescriptorBufferInfo pointInfo{ vk_point_light_ssbo, 0, sizeof(PointLight) * MAX_POINT_LIGHTS };
                VkDescriptorBufferInfo spotInfo{ vk_spot_light_ssbo, 0, sizeof(Spotlight) * MAX_SPOT_LIGHTS };

                VkWriteDescriptorSet w[3]{};
                w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[0].dstSet = vk_scene_set;
                w[0].dstBinding = 0;
                w[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                w[0].descriptorCount = 1;
                w[0].pBufferInfo = &sceneInfo;

                w[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[1].dstSet = vk_scene_set;
                w[1].dstBinding = 1;
                w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[1].descriptorCount = 1;
                w[1].pBufferInfo = &pointInfo;

                w[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[2].dstSet = vk_scene_set;
                w[2].dstBinding = 2;
                w[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[2].descriptorCount = 1;
                w[2].pBufferInfo = &spotInfo;

                vkUpdateDescriptorSets(device, 3, w, 0, nullptr);
            }

            // Model sets
            {
                VkDescriptorSetLayout layouts[2] = { vk_model_set_layout, vk_model_set_layout };
                VkDescriptorSetAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool = vk_descriptor_pool;
                ai.descriptorSetCount = 2;
                ai.pSetLayouts = layouts;
                if (vkAllocateDescriptorSets(device, &ai, vk_model_sets) != VK_SUCCESS) {
                    std::cerr << "Failed to allocate model sets\n";
                    return false;
                }

                for (int i = 0; i < 2; ++i) {
                    VkDescriptorBufferInfo mInfo{ vk_model_ubos[i], 0, sizeof(ModelUniform) };
                    VkWriteDescriptorSet w{};
                    w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                    w.dstSet = vk_model_sets[i];
                    w.dstBinding = 0;
                    w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    w.descriptorCount = 1;
                    w.pBufferInfo = &mInfo;
                    vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
                }
            }

            return true;
        }

        // ============================================================
        //  Обновление
        // ============================================================
        void updateSceneUniforms() {
            auto& ctx = graphics::internal::context;

            float aspect = static_cast<float>(ctx.swapchain_extent.width) /
                static_cast<float>(ctx.swapchain_extent.height);

            float yaw = glm::radians(camera_yaw);
            float pitch = glm::radians(glm::clamp(camera_pitch, -85.0f, 85.0f));
            glm::vec3 target(0.0f);
            glm::vec3 camPos = target + camera_distance * glm::vec3(
                cosf(pitch) * sinf(yaw),
                sinf(pitch),
                cosf(pitch) * cosf(yaw));

            glm::mat4 view = glm::lookAt(camPos, target, glm::vec3(0, 1, 0));
            glm::mat4 proj = glm::perspective(glm::radians(camera_fov), aspect, 0.1f, 100.0f);
            proj[1][1] *= -1.0f;

            SceneUniforms su{};
            su.view_projection = proj * view;
            su.camera_position = camPos;
            su.directional_light_direction = directional_direction;
            su.directional_light_color = directional_color;
            su.ambient_light = ambient_light;
            su.point_light_count = static_cast<uint32_t>(point_lights.size());
            su.spot_light_count = static_cast<uint32_t>(spot_lights.size());
            std::memcpy(scene_ubo_mapped, &su, sizeof(SceneUniforms));
            flushBuffer(vma_scene_ubo);

            if (!point_lights.empty()) {
                std::memcpy(point_light_ssbo_mapped, point_lights.data(),
                    sizeof(PointLight) * point_lights.size());
                flushBuffer(vma_point_light_ssbo);
            }
            if (!spot_lights.empty()) {
                std::memcpy(spot_light_ssbo_mapped, spot_lights.data(),
                    sizeof(Spotlight) * spot_lights.size());
                flushBuffer(vma_spot_light_ssbo);
            }
        }

        void updateModelUniform(int index, const ObjectState& o) {
            glm::mat4 model(1.0f);
            model = glm::translate(model, o.position);
            model = glm::rotate(model, glm::radians(o.rotation.x), glm::vec3(1, 0, 0));
            model = glm::rotate(model, glm::radians(o.rotation.y), glm::vec3(0, 1, 0));
            model = glm::rotate(model, glm::radians(o.rotation.z), glm::vec3(0, 0, 1));
            model = glm::scale(model, o.scale);

            ModelUniform mu{};
            mu.model = model;
            mu.albedo = o.albedo;
            mu.specular = o.specular;
            mu.alpha = o.alpha;
            std::memcpy(model_ubo_mapped[index], &mu, sizeof(mu));
            flushBuffer(vma_model_ubos[index]);
        }

    } // namespace

    // ============================================================
    //  INIT / SHUTDOWN
    // ============================================================
    bool initialize() {
        auto vertices = generateCylinder(1.0f, 2.0f, 50);
        auto indices = generateIndices(static_cast<uint32_t>(vertices.size()));
        vertex_count = static_cast<uint32_t>(vertices.size());
        index_count = static_cast<uint32_t>(indices.size());

        {
            void* data = nullptr;
            if (!createBuffer(sizeof(Vertex) * vertex_count, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                vk_vertex_buffer, vma_vertex_buffer, &data)) return false;
            std::memcpy(data, vertices.data(), sizeof(Vertex) * vertex_count);
        }
        {
            void* data = nullptr;
            if (!createBuffer(sizeof(uint32_t) * index_count, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                vk_index_buffer, vma_index_buffer, &data)) return false;
            std::memcpy(data, indices.data(), sizeof(uint32_t) * index_count);
        }

        if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk_scene_ubo, vma_scene_ubo, &scene_ubo_mapped)) return false;

        if (!createBuffer(sizeof(PointLight) * MAX_POINT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vk_point_light_ssbo, vma_point_light_ssbo, &point_light_ssbo_mapped)) return false;

        if (!createBuffer(sizeof(Spotlight) * MAX_SPOT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vk_spot_light_ssbo, vma_spot_light_ssbo, &spot_light_ssbo_mapped)) return false;

        for (int i = 0; i < 2; ++i) {
            if (!createBuffer(sizeof(ModelUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                vk_model_ubos[i], vma_model_ubos[i], &model_ubo_mapped[i])) return false;
        }

        if (!createGraphicsPipeline()) return false;
        if (!createDescriptorSets()) return false;

        obj1.position = glm::vec3(1.8f, 0.0f, 0.0f);
        obj1.albedo = glm::vec3(1.0f, 0.6f, 0.3f);
        obj1.specular = glm::vec3(1.0f);
        obj1.alpha = 64.0f;

        obj2.position = glm::vec3(-1.8f, 0.0f, 0.0f);
        obj2.albedo = glm::vec3(0.3f, 0.6f, 1.0f);
        obj2.specular = glm::vec3(0.8f);
        obj2.alpha = 32.0f;

        directional_direction = glm::vec3(-0.3f, -1.0f, -0.5f);
        directional_color = glm::vec3(1.5f);

        {
            PointLight pl{};
            pl.position = glm::vec3(0.0f, 2.5f, 2.0f);
            pl.color = glm::vec3(5.0f);
            point_lights.push_back(pl);
        }

        std::cout << "Lab 2 initialized:\n";
        std::cout << "  sizeof(SceneUniforms) = " << sizeof(SceneUniforms) << "\n";
        std::cout << "  sizeof(ModelUniform)  = " << sizeof(ModelUniform) << "\n";
        std::cout << "  sizeof(PointLight)    = " << sizeof(PointLight) << "\n";
        std::cout << "  sizeof(Spotlight)     = " << sizeof(Spotlight) << "\n";

        return true;
    }

    void shutdown() {
        auto& ctx = graphics::internal::context;
        vkDeviceWaitIdle(ctx.device);

        if (vk_descriptor_pool)      vkDestroyDescriptorPool(ctx.device, vk_descriptor_pool, nullptr);
        if (vk_scene_set_layout)     vkDestroyDescriptorSetLayout(ctx.device, vk_scene_set_layout, nullptr);
        if (vk_model_set_layout)     vkDestroyDescriptorSetLayout(ctx.device, vk_model_set_layout, nullptr);
        if (vk_pipeline)             vkDestroyPipeline(ctx.device, vk_pipeline, nullptr);
        if (vk_pipeline_layout)      vkDestroyPipelineLayout(ctx.device, vk_pipeline_layout, nullptr);

        if (vk_vertex_buffer)        vmaDestroyBuffer(ctx.allocator, vk_vertex_buffer, vma_vertex_buffer);
        if (vk_index_buffer)         vmaDestroyBuffer(ctx.allocator, vk_index_buffer, vma_index_buffer);
        if (vk_scene_ubo)            vmaDestroyBuffer(ctx.allocator, vk_scene_ubo, vma_scene_ubo);
        if (vk_point_light_ssbo)     vmaDestroyBuffer(ctx.allocator, vk_point_light_ssbo, vma_point_light_ssbo);
        if (vk_spot_light_ssbo)      vmaDestroyBuffer(ctx.allocator, vk_spot_light_ssbo, vma_spot_light_ssbo);

        for (int i = 0; i < 2; ++i) {
            if (vk_model_ubos[i])    vmaDestroyBuffer(ctx.allocator, vk_model_ubos[i], vma_model_ubos[i]);
        }

        vkQueueWaitIdle(ctx.graphics_queue);
    }

    // ============================================================
    //  UPDATE
    // ============================================================
    void update(double /*time*/) {
        ImGui::Begin("Lab 2: Lighting");

        if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::SliderFloat("Yaw", &camera_yaw, -180.0f, 180.0f, "%.1f");
            ImGui::SliderFloat("Pitch", &camera_pitch, -85.0f, 85.0f, "%.1f");
            ImGui::SliderFloat("Distance", &camera_distance, 1.5f, 30.0f);
            ImGui::SliderFloat("FOV", &camera_fov, 20.0f, 100.0f);
            if (ImGui::Button("Reset camera")) {
                camera_yaw = 0.0f;
                camera_pitch = 20.0f;
                camera_distance = 6.0f;
                camera_fov = 45.0f;
            }
        }

        if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::ColorEdit3("Ambient light", &ambient_light.x);
        }

        if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::DragFloat3("Direction", &directional_direction.x, 0.02f, -1.0f, 1.0f);
            ImGui::ColorEdit3("Color", &directional_color.x);
        }

        auto drawObjectUI = [&](const char* name, ObjectState& o) {
            if (ImGui::CollapsingHeader(name, ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID(name);
                ImGui::DragFloat3("Position", &o.position.x, 0.05f);
                ImGui::DragFloat3("Rotation", &o.rotation.x, 1.0f);
                ImGui::DragFloat3("Scale", &o.scale.x, 0.05f, 0.1f, 5.0f);
                ImGui::ColorEdit3("Albedo", &o.albedo.x);
                ImGui::ColorEdit3("Specular", &o.specular.x);
                ImGui::SliderFloat("Alpha (shininess)", &o.alpha, 1.0f, 256.0f);
                ImGui::PopID();
            }
            };
        drawObjectUI("Object 1", obj1);
        drawObjectUI("Object 2", obj2);

        if (ImGui::CollapsingHeader("Point lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Add point light") && point_lights.size() < MAX_POINT_LIGHTS) {
                PointLight pl{};
                pl.position = glm::vec3(0.0f, 2.0f, 3.0f);
                pl.color = glm::vec3(5.0f);
                point_lights.push_back(pl);
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear point lights")) point_lights.clear();

            for (size_t i = 0; i < point_lights.size(); ++i) {
                auto& pl = point_lights[i];
                ImGui::PushID(static_cast<int>(i) + 1000);
                if (ImGui::CollapsingHeader(("Point #" + std::to_string(i)).c_str())) {
                    ImGui::DragFloat3("Position", &pl.position.x, 0.05f);
                    ImGui::ColorEdit3("Color", &pl.color.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
                    if (ImGui::Button("Remove")) {
                        point_lights.erase(point_lights.begin() + i);
                        ImGui::PopID();
                        break;
                    }
                }
                ImGui::PopID();
            }
        }

        if (ImGui::CollapsingHeader("Spot lights", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Button("Add spot light") && spot_lights.size() < MAX_SPOT_LIGHTS) {
                Spotlight sl{};
                sl.position = glm::vec3(0.0f, 3.0f, 3.0f);
                sl.color = glm::vec3(5.0f);
                sl.direction = glm::vec3(0.0f, -1.0f, 0.0f);
                sl.inner_angle = 15.0f;
                sl.outer_angle = 25.0f;
                spot_lights.push_back(sl);
            }
            ImGui::SameLine();
            if (ImGui::Button("Clear spot lights")) spot_lights.clear();

            for (size_t i = 0; i < spot_lights.size(); ++i) {
                auto& sl = spot_lights[i];
                ImGui::PushID(static_cast<int>(i) + 2000);
                if (ImGui::CollapsingHeader(("Spot #" + std::to_string(i)).c_str())) {
                    ImGui::DragFloat3("Position", &sl.position.x, 0.05f);
                    ImGui::ColorEdit3("Color", &sl.color.x, ImGuiColorEditFlags_HDR | ImGuiColorEditFlags_Float);
                    ImGui::DragFloat3("Direction", &sl.direction.x, 0.02f, -1.0f, 1.0f);
                    ImGui::SliderFloat("Inner angle (deg)", &sl.inner_angle, 0.0f, 90.0f);
                    ImGui::SliderFloat("Outer angle (deg)", &sl.outer_angle, 0.0f, 90.0f);
                    if (ImGui::Button("Remove")) {
                        spot_lights.erase(spot_lights.begin() + i);
                        ImGui::PopID();
                        break;
                    }
                }
                ImGui::PopID();
            }
        }

        ImGui::End();

        updateSceneUniforms();
        updateModelUniform(0, obj1);
        updateModelUniform(1, obj2);
    }

    // ============================================================
    //  RENDER
    // ============================================================
    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;

        VkCommandBuffer cmd = fd.command_buffer;
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkClearValue clearValues[2]{};
        clearValues[0].color = { { 0.05f, 0.05f, 0.08f, 1.0f } };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo rpBegin{};
        rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rpBegin.renderPass = ctx.render_pass;
        rpBegin.framebuffer = fd.framebuffer;
        rpBegin.renderArea = { { 0, 0 }, ctx.swapchain_extent };
        rpBegin.clearValueCount = 2;
        rpBegin.pClearValues = clearValues;
        vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{ 0, 0,
            static_cast<float>(ctx.swapchain_extent.width),
            static_cast<float>(ctx.swapchain_extent.height),
            0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{ { 0, 0 }, ctx.swapchain_extent };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline);

        VkBuffer vertexBuffers[] = { vk_vertex_buffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, vertexBuffers, offsets);
        vkCmdBindIndexBuffer(cmd, vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
            0, 1, &vk_scene_set, 0, nullptr);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
            1, 1, &vk_model_sets[0], 0, nullptr);
        vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
            1, 1, &vk_model_sets[1], 0, nullptr);
        vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace application