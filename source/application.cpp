#include "application.hpp"

#include <imgui.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "lodepng.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <functional>

namespace application {

    namespace {

        constexpr int MAX_POINT_LIGHTS = 16;
        constexpr int MAX_SPOT_LIGHTS = 16;
        constexpr int MAX_OBJECTS = 2;
        constexpr int MAX_TEXTURES = 16;

        // ============================================================
        //  Структуры
        // ============================================================

        struct Vertex {
            glm::vec3 pos;
            glm::vec3 color;
            glm::vec3 normal;
            glm::vec2 texCoord;

            static VkVertexInputBindingDescription bindingDescription() {
                return { 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
            }

            static std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions() {
                return { {
                    { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                    { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) },
                    { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                    { 3, 0, VK_FORMAT_R32G32_SFLOAT,    offsetof(Vertex, texCoord) },
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

        struct alignas(16) Spotlight {
            glm::vec3 position;
            float     _pad0 = 0;
            glm::vec3 color;
            float     _pad1 = 0;
            glm::vec3 direction;
            float     inner_angle = 15.0f;
            float     outer_angle = 25.0f;
            float     _pad2[2] = { 0, 0 };
        };

        static_assert(sizeof(PointLight) == 32, "PointLight must be 32 bytes");
        static_assert(sizeof(Spotlight) == 64, "Spotlight must be 64 bytes");

        struct PushConstants {
            int albedoIndex = 0;
            int specularIndex = -1;
            int emissiveIndex = -1;
            int useNontrivialSampling = 0;
            float time = 0.0f;
        };

        struct Texture {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE;
            VkSampler sampler = VK_NULL_HANDLE;
            std::string name;
        };

        // ============================================================
        //  Vulkan-объекты
        // ============================================================

        VkPipelineLayout vk_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline       vk_pipeline = VK_NULL_HANDLE;

        VkDescriptorSetLayout vk_scene_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout vk_model_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout vk_texture_set_layout = VK_NULL_HANDLE;

        VkDescriptorPool vk_descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet  vk_scene_set = VK_NULL_HANDLE;
        VkDescriptorSet  vk_model_set = VK_NULL_HANDLE;
        VkDescriptorSet  vk_texture_set = VK_NULL_HANDLE;

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

        VkBuffer      vk_model_ubo = VK_NULL_HANDLE;
        VmaAllocation vma_model_ubo = VK_NULL_HANDLE;
        void* model_ubo_mapped = nullptr;
        VkDeviceSize  g_model_ubo_stride = 0;

        std::vector<Texture> g_albedoTextures;
        std::vector<Texture> g_specularTextures;
        std::vector<Texture> g_emissiveTextures;
        Texture g_defaultBlack;

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

            int albedoIndex = 0;
            int specularIndex = -1;
            int emissiveIndex = -1;
            bool useNontrivialSampling = false;
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
        std::vector<Spotlight> spot_lights;

        // ============================================================
        //  Вспомогательные функции
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

        void executeOneShotCommands(std::function<void(VkCommandBuffer)>&& recorder) {
            VkDevice device = graphics::internal::context.device;
            VkCommandPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            poolInfo.queueFamilyIndex = graphics::internal::context.graphics_queue_index;
            poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            VkCommandPool commandPool;
            vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

            VkCommandBufferAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocInfo.commandPool = commandPool;
            allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocInfo.commandBufferCount = 1;
            VkCommandBuffer cmd;
            vkAllocateCommandBuffers(device, &allocInfo, &cmd);

            VkCommandBufferBeginInfo beginInfo{};
            beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            recorder(cmd);

            vkEndCommandBuffer(cmd);

            VkSubmitInfo submitInfo{};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;

            vkQueueSubmit(graphics::internal::context.graphics_queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(graphics::internal::context.graphics_queue);

            vkDestroyCommandPool(device, commandPool, nullptr);
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

            if (mapped) *mapped = info.pMappedData;
            return true;
        }

        void createBufferOnGPU(VkDeviceSize size, VkBufferUsageFlags usage,
            VkBuffer& buffer, VmaAllocation& allocation, const void* data) {
            VkBuffer stagingBuffer;
            VmaAllocation stagingAllocation;
            void* stagingData;
            VkBufferCreateInfo stagingInfo{};
            stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            stagingInfo.size = size;
            stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo stagingAllocInfo{};
            stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
            stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            vmaCreateBuffer(graphics::internal::context.allocator, &stagingInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, nullptr);
            vmaMapMemory(graphics::internal::context.allocator, stagingAllocation, &stagingData);
            memcpy(stagingData, data, static_cast<size_t>(size));
            vmaUnmapMemory(graphics::internal::context.allocator, stagingAllocation);

            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = size;
            bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &allocInfo, &buffer, &allocation, nullptr);

            executeOneShotCommands([&](VkCommandBuffer cmd) {
                VkBufferCopy copyRegion{};
                copyRegion.size = size;
                vkCmdCopyBuffer(cmd, stagingBuffer, buffer, 1, &copyRegion);
                });

            vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
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

        bool createTextureFromPixels(const std::vector<unsigned char>& pixels, unsigned width, unsigned height, Texture& outTexture) {
            VkDeviceSize imageSize = width * height * 4;

            VkImageCreateInfo imageInfo{};
            imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType = VK_IMAGE_TYPE_2D;
            imageInfo.extent.width = width;
            imageInfo.extent.height = height;
            imageInfo.extent.depth = 1;
            imageInfo.mipLevels = 1;
            imageInfo.arrayLayers = 1;
            imageInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
            imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
            imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo allocInfo{};
            allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
            if (vmaCreateImage(graphics::internal::context.allocator, &imageInfo, &allocInfo,
                &outTexture.image, &outTexture.allocation, nullptr) != VK_SUCCESS) return false;

            VkBuffer stagingBuffer;
            VmaAllocation stagingAllocation;
            void* stagingData;
            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = imageSize;
            bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VmaAllocationCreateInfo stagingAllocInfo{};
            stagingAllocInfo.usage = VMA_MEMORY_USAGE_AUTO;
            stagingAllocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
            vmaCreateBuffer(graphics::internal::context.allocator, &bufferInfo, &stagingAllocInfo, &stagingBuffer, &stagingAllocation, nullptr);
            vmaMapMemory(graphics::internal::context.allocator, stagingAllocation, &stagingData);
            memcpy(stagingData, pixels.data(), static_cast<size_t>(imageSize));
            vmaUnmapMemory(graphics::internal::context.allocator, stagingAllocation);

            executeOneShotCommands([&](VkCommandBuffer cmd) {
                VkImageMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.image = outTexture.image;
                barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                barrier.subresourceRange.baseMipLevel = 0;
                barrier.subresourceRange.levelCount = 1;
                barrier.subresourceRange.baseArrayLayer = 0;
                barrier.subresourceRange.layerCount = 1;
                barrier.srcAccessMask = 0;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &barrier);

                VkBufferImageCopy region{};
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.mipLevel = 0;
                region.imageSubresource.baseArrayLayer = 0;
                region.imageSubresource.layerCount = 1;
                region.imageOffset = { 0, 0, 0 };
                region.imageExtent = { width, height, 1 };
                vkCmdCopyBufferToImage(cmd, stagingBuffer, outTexture.image,
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

                VkImageMemoryBarrier barrier2 = barrier;
                barrier2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                barrier2.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                barrier2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                    0, 0, nullptr, 0, nullptr, 1, &barrier2);
                });

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image = outTexture.image;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = VK_FORMAT_R8G8B8A8_SRGB;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;
            vkCreateImageView(graphics::internal::context.device, &viewInfo, nullptr, &outTexture.view);

            VkSamplerCreateInfo samplerInfo{};
            samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
            samplerInfo.magFilter = VK_FILTER_LINEAR;
            samplerInfo.minFilter = VK_FILTER_LINEAR;
            samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            samplerInfo.anisotropyEnable = VK_FALSE;
            samplerInfo.maxAnisotropy = 1.0f;
            samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
            samplerInfo.unnormalizedCoordinates = VK_FALSE;
            samplerInfo.compareEnable = VK_FALSE;
            samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
            samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
            vkCreateSampler(graphics::internal::context.device, &samplerInfo, nullptr, &outTexture.sampler);

            vmaDestroyBuffer(graphics::internal::context.allocator, stagingBuffer, stagingAllocation);
            return true;
        }

        bool createTexture(const std::string& path, Texture& outTexture) {
            std::vector<unsigned char> image;
            unsigned width, height;
            unsigned error = lodepng::decode(image, width, height, path);
            if (error) {
                std::cerr << "Failed to load texture: " << path << " (" << lodepng_error_text(error) << ")\n";
                return false;
            }
            outTexture.name = path;
            return createTextureFromPixels(image, width, height, outTexture);
        }

        bool createDefaultBlackTexture(Texture& outTexture) {
            std::vector<unsigned char> pixel = { 0, 0, 0, 255 };
            outTexture.name = "default_black";
            return createTextureFromPixels(pixel, 1, 1, outTexture);
        }

        std::vector<Vertex> generateCylinder(float radius, float height, int segments) {
            std::vector<Vertex> vertices;
            const float halfH = height / 2.0f;
            const float PI = 3.14159265f;

            auto makeVertex = [&](float x, float y, float z, float nx, float ny, float nz, float u, float v) -> Vertex {
                float t = (y + halfH) / height;
                glm::vec3 color(t, 1.0f - t, 0.5f);
                return { glm::vec3(x, y, z), color, glm::vec3(nx, ny, nz), glm::vec2(u, v) };
                };

            for (int i = 0; i < segments; ++i) {
                float a0 = 2.0f * PI * i / segments;
                float a1 = 2.0f * PI * (i + 1) / segments;
                float c0 = cosf(a0), s0 = sinf(a0);
                float c1 = cosf(a1), s1 = sinf(a1);
                float x0 = radius * c0, z0 = radius * s0;
                float x1 = radius * c1, z1 = radius * s1;
                float u0 = (float)i / segments;
                float u1 = (float)(i + 1) / segments;

                vertices.push_back(makeVertex(x0, -halfH, z0, c0, 0.0f, s0, u0, 0.0f));
                vertices.push_back(makeVertex(x1, -halfH, z1, c1, 0.0f, s1, u1, 0.0f));
                vertices.push_back(makeVertex(x1, halfH, z1, c1, 0.0f, s1, u1, 1.0f));

                vertices.push_back(makeVertex(x0, -halfH, z0, c0, 0.0f, s0, u0, 0.0f));
                vertices.push_back(makeVertex(x1, halfH, z1, c1, 0.0f, s1, u1, 1.0f));
                vertices.push_back(makeVertex(x0, halfH, z0, c0, 0.0f, s0, u0, 1.0f));
            }

            for (int i = 0; i < segments; ++i) {
                float a0 = 2.0f * PI * i / segments;
                float a1 = 2.0f * PI * (i + 1) / segments;
                vertices.push_back(makeVertex(0.0f, -halfH, 0.0f, 0.0f, -1.0f, 0.0f, 0.5f, 0.5f));
                vertices.push_back(makeVertex(radius * cosf(a1), -halfH, radius * sinf(a1), 0.0f, -1.0f, 0.0f,
                    0.5f + 0.5f * cosf(a1), 0.5f + 0.5f * sinf(a1)));
                vertices.push_back(makeVertex(radius * cosf(a0), -halfH, radius * sinf(a0), 0.0f, -1.0f, 0.0f,
                    0.5f + 0.5f * cosf(a0), 0.5f + 0.5f * sinf(a0)));
            }
            for (int i = 0; i < segments; ++i) {
                float a0 = 2.0f * PI * i / segments;
                float a1 = 2.0f * PI * (i + 1) / segments;
                vertices.push_back(makeVertex(0.0f, halfH, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f, 0.5f));
                vertices.push_back(makeVertex(radius * cosf(a0), halfH, radius * sinf(a0), 0.0f, 1.0f, 0.0f,
                    0.5f + 0.5f * cosf(a0), 0.5f + 0.5f * sinf(a0)));
                vertices.push_back(makeVertex(radius * cosf(a1), halfH, radius * sinf(a1), 0.0f, 1.0f, 0.0f,
                    0.5f + 0.5f * cosf(a1), 0.5f + 0.5f * sinf(a1)));
            }

            return vertices;
        }

        std::vector<uint32_t> generateIndices(uint32_t count) {
            std::vector<uint32_t> indices(count);
            for (uint32_t i = 0; i < count; ++i) indices[i] = i;
            return indices;
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

            // Set 0: Scene
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

            // Set 1: Model (dynamic UBO)
            {
                VkDescriptorSetLayoutBinding b{};
                b.binding = 0;
                b.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
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

            // Set 2: Textures
            {
                VkDescriptorSetLayoutBinding b[3]{};
                b[0].binding = 0;
                b[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                b[0].descriptorCount = MAX_TEXTURES;
                b[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                b[1].binding = 1;
                b[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                b[1].descriptorCount = MAX_TEXTURES;
                b[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
                b[2].binding = 2;
                b[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                b[2].descriptorCount = MAX_TEXTURES;
                b[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

                VkDescriptorSetLayoutCreateInfo info{};
                info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
                info.bindingCount = 3;
                info.pBindings = b;
                if (vkCreateDescriptorSetLayout(device, &info, nullptr, &vk_texture_set_layout) != VK_SUCCESS) {
                    std::cerr << "Failed to create texture set layout\n";
                    return false;
                }
            }

            VkDescriptorSetLayout setLayouts[3] = { vk_scene_set_layout, vk_model_set_layout, vk_texture_set_layout };

            VkPushConstantRange pushRange{};
            pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            pushRange.offset = 0;
            pushRange.size = sizeof(PushConstants);

            VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
            pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipelineLayoutInfo.setLayoutCount = 3;
            pipelineLayoutInfo.pSetLayouts = setLayouts;
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushRange;

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

        bool createDescriptorSets() {
            VkDevice device = graphics::internal::context.device;

            VkDescriptorPoolSize poolSizes[4]{};
            poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            poolSizes[0].descriptorCount = 1;
            poolSizes[1].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            poolSizes[1].descriptorCount = 1;
            poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSizes[2].descriptorCount = 2;
            poolSizes[3].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            poolSizes[3].descriptorCount = 3 * MAX_TEXTURES;

            VkDescriptorPoolCreateInfo poolInfo{};
            poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            poolInfo.poolSizeCount = 4;
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

            // Model set (dynamic)
            {
                VkDescriptorSetAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool = vk_descriptor_pool;
                ai.descriptorSetCount = 1;
                ai.pSetLayouts = &vk_model_set_layout;
                if (vkAllocateDescriptorSets(device, &ai, &vk_model_set) != VK_SUCCESS) {
                    std::cerr << "Failed to allocate model set\n";
                    return false;
                }

                VkDescriptorBufferInfo mInfo{ vk_model_ubo, 0, sizeof(ModelUniform) };
                VkWriteDescriptorSet w{};
                w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w.dstSet = vk_model_set;
                w.dstBinding = 0;
                w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                w.descriptorCount = 1;
                w.pBufferInfo = &mInfo;
                vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
            }

            // Texture set
            {
                VkDescriptorSetAllocateInfo ai{};
                ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
                ai.descriptorPool = vk_descriptor_pool;
                ai.descriptorSetCount = 1;
                ai.pSetLayouts = &vk_texture_set_layout;
                if (vkAllocateDescriptorSets(device, &ai, &vk_texture_set) != VK_SUCCESS) {
                    std::cerr << "Failed to allocate texture set\n";
                    return false;
                }

                std::vector<VkDescriptorImageInfo> albedoInfos(MAX_TEXTURES);
                std::vector<VkDescriptorImageInfo> specularInfos(MAX_TEXTURES);
                std::vector<VkDescriptorImageInfo> emissiveInfos(MAX_TEXTURES);

                auto fillInfos = [&](std::vector<VkDescriptorImageInfo>& infos, std::vector<Texture>& textures) {
                    for (int i = 0; i < MAX_TEXTURES; ++i) {
                        if (i < (int)textures.size() && textures[i].view != VK_NULL_HANDLE) {
                            infos[i].sampler = textures[i].sampler;
                            infos[i].imageView = textures[i].view;
                        }
                        else {
                            infos[i].sampler = g_defaultBlack.sampler;
                            infos[i].imageView = g_defaultBlack.view;
                        }
                        infos[i].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    }
                    };

                fillInfos(albedoInfos, g_albedoTextures);
                fillInfos(specularInfos, g_specularTextures);
                fillInfos(emissiveInfos, g_emissiveTextures);

                VkWriteDescriptorSet writes[3]{};
                writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[0].dstSet = vk_texture_set;
                writes[0].dstBinding = 0;
                writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[0].descriptorCount = MAX_TEXTURES;
                writes[0].pImageInfo = albedoInfos.data();
                writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[1].dstSet = vk_texture_set;
                writes[1].dstBinding = 1;
                writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[1].descriptorCount = MAX_TEXTURES;
                writes[1].pImageInfo = specularInfos.data();
                writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[2].dstSet = vk_texture_set;
                writes[2].dstBinding = 2;
                writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[2].descriptorCount = MAX_TEXTURES;
                writes[2].pImageInfo = emissiveInfos.data();

                vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
            }

            return true;
        }

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
                std::memcpy(point_light_ssbo_mapped, point_lights.data(), sizeof(PointLight) * point_lights.size());
                flushBuffer(vma_point_light_ssbo);
            }
            if (!spot_lights.empty()) {
                std::memcpy(spot_light_ssbo_mapped, spot_lights.data(), sizeof(Spotlight) * spot_lights.size());
                flushBuffer(vma_spot_light_ssbo);
            }
        }

        void updateModelUniforms() {
            uint8_t* base = (uint8_t*)model_ubo_mapped;

            auto fill = [&](int index, const ObjectState& o) {
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
                memcpy(base + index * g_model_ubo_stride, &mu, sizeof(ModelUniform));
                };

            fill(0, obj1);
            fill(1, obj2);
            flushBuffer(vma_model_ubo);
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

        createBufferOnGPU(sizeof(Vertex) * vertex_count, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vk_vertex_buffer, vma_vertex_buffer, vertices.data());
        createBufferOnGPU(sizeof(uint32_t) * index_count, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            vk_index_buffer, vma_index_buffer, indices.data());

        if (!createBuffer(sizeof(SceneUniforms), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk_scene_ubo, vma_scene_ubo, &scene_ubo_mapped)) return false;
        if (!createBuffer(sizeof(PointLight) * MAX_POINT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vk_point_light_ssbo, vma_point_light_ssbo, &point_light_ssbo_mapped)) return false;
        if (!createBuffer(sizeof(Spotlight) * MAX_SPOT_LIGHTS, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vk_spot_light_ssbo, vma_spot_light_ssbo, &spot_light_ssbo_mapped)) return false;
        {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(graphics::internal::context.physical_device, &props);
            VkDeviceSize align = props.limits.minUniformBufferOffsetAlignment;
            g_model_ubo_stride = (sizeof(ModelUniform) + align - 1) / align * align;
        }
        if (!createBuffer(g_model_ubo_stride * MAX_OBJECTS, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk_model_ubo, vma_model_ubo, &model_ubo_mapped)) return false;

        // Заглушка
        if (!createDefaultBlackTexture(g_defaultBlack)) {
            std::cerr << "Failed to create default black texture\n";
            return false;
        }

        // Загрузка текстур (если файлов нет — просто пропустим)
        g_albedoTextures.resize(2);
        if (!createTexture("textures/brick.png", g_albedoTextures[0])) {
            g_albedoTextures[0] = g_defaultBlack;
        }
        if (!createTexture("textures/wood.png", g_albedoTextures[1])) {
            g_albedoTextures[1] = g_defaultBlack;
        }

        g_specularTextures.resize(1);
        if (!createTexture("textures/specular_map.png", g_specularTextures[0])) {
            g_specularTextures.clear();
        }

        g_emissiveTextures.resize(1);
        if (!createTexture("textures/emissive_map.png", g_emissiveTextures[0])) {
            g_emissiveTextures.clear();
        }

        if (!createGraphicsPipeline()) return false;
        if (!createDescriptorSets()) return false;

        obj1.position = glm::vec3(1.8f, 0.0f, 0.0f);
        obj1.albedo = glm::vec3(1.0f, 0.6f, 0.3f);
        obj1.specular = glm::vec3(1.0f);
        obj1.alpha = 64.0f;
        obj1.albedoIndex = 0;
        obj1.specularIndex = -1;
        obj1.emissiveIndex = -1;

        obj2.position = glm::vec3(-1.8f, 0.0f, 0.0f);
        obj2.albedo = glm::vec3(0.3f, 0.6f, 1.0f);
        obj2.specular = glm::vec3(0.8f);
        obj2.alpha = 32.0f;
        obj2.albedoIndex = 1;
        obj2.specularIndex = -1;
        obj2.emissiveIndex = -1;

        directional_direction = glm::vec3(-0.3f, -1.0f, -0.5f);
        directional_color = glm::vec3(1.5f);

        PointLight pl{};
        pl.position = glm::vec3(0.0f, 2.5f, 2.0f);
        pl.color = glm::vec3(5.0f);
        point_lights.push_back(pl);

        return true;
    }

    void shutdown() {
        auto& ctx = graphics::internal::context;
        vkDeviceWaitIdle(ctx.device);

        if (vk_descriptor_pool)    vkDestroyDescriptorPool(ctx.device, vk_descriptor_pool, nullptr);
        if (vk_scene_set_layout)   vkDestroyDescriptorSetLayout(ctx.device, vk_scene_set_layout, nullptr);
        if (vk_model_set_layout)   vkDestroyDescriptorSetLayout(ctx.device, vk_model_set_layout, nullptr);
        if (vk_texture_set_layout) vkDestroyDescriptorSetLayout(ctx.device, vk_texture_set_layout, nullptr);
        if (vk_pipeline)           vkDestroyPipeline(ctx.device, vk_pipeline, nullptr);
        if (vk_pipeline_layout)    vkDestroyPipelineLayout(ctx.device, vk_pipeline_layout, nullptr);

        if (vk_vertex_buffer)      vmaDestroyBuffer(ctx.allocator, vk_vertex_buffer, vma_vertex_buffer);
        if (vk_index_buffer)       vmaDestroyBuffer(ctx.allocator, vk_index_buffer, vma_index_buffer);
        if (vk_scene_ubo)          vmaDestroyBuffer(ctx.allocator, vk_scene_ubo, vma_scene_ubo);
        if (vk_point_light_ssbo)   vmaDestroyBuffer(ctx.allocator, vk_point_light_ssbo, vma_point_light_ssbo);
        if (vk_spot_light_ssbo)    vmaDestroyBuffer(ctx.allocator, vk_spot_light_ssbo, vma_spot_light_ssbo);
        if (vk_model_ubo)          vmaDestroyBuffer(ctx.allocator, vk_model_ubo, vma_model_ubo);

        auto destroyTex = [&](Texture& t) {
            if (t.sampler) vkDestroySampler(ctx.device, t.sampler, nullptr);
            if (t.view)    vkDestroyImageView(ctx.device, t.view, nullptr);
            if (t.image)   vmaDestroyImage(ctx.allocator, t.image, t.allocation);
            };

        for (auto& t : g_albedoTextures)    if (t.image != g_defaultBlack.image) destroyTex(t);
        for (auto& t : g_specularTextures)  destroyTex(t);
        for (auto& t : g_emissiveTextures)  destroyTex(t);
        destroyTex(g_defaultBlack);

        vkQueueWaitIdle(ctx.graphics_queue);
    }

    // ============================================================
    //  UPDATE
    // ============================================================

    void update(double) {
        ImGui::Begin("Lab 3: Textures");

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

        if (ImGui::CollapsingHeader("Ambient", ImGuiTreeNodeFlags_DefaultOpen))
            ImGui::ColorEdit3("Ambient light", &ambient_light.x);

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

                if (ImGui::BeginCombo("Albedo Texture", g_albedoTextures[o.albedoIndex].name.c_str())) {
                    for (int i = 0; i < (int)g_albedoTextures.size(); ++i) {
                        bool sel = (o.albedoIndex == i);
                        if (ImGui::Selectable(g_albedoTextures[i].name.c_str(), sel)) o.albedoIndex = i;
                        if (sel) ImGui::SetItemDefaultFocus();
                    }
                    ImGui::EndCombo();
                }

                const char* specPreview = (o.specularIndex < 0 || g_specularTextures.empty())
                    ? "None" : g_specularTextures[o.specularIndex].name.c_str();
                if (ImGui::BeginCombo("Specular Texture", specPreview)) {
                    if (ImGui::Selectable("None", o.specularIndex < 0)) o.specularIndex = -1;
                    for (int i = 0; i < (int)g_specularTextures.size(); ++i) {
                        bool sel = (o.specularIndex == i);
                        if (ImGui::Selectable(g_specularTextures[i].name.c_str(), sel)) o.specularIndex = i;
                    }
                    ImGui::EndCombo();
                }

                const char* emPreview = (o.emissiveIndex < 0 || g_emissiveTextures.empty())
                    ? "None" : g_emissiveTextures[o.emissiveIndex].name.c_str();
                if (ImGui::BeginCombo("Emissive Texture", emPreview)) {
                    if (ImGui::Selectable("None", o.emissiveIndex < 0)) o.emissiveIndex = -1;
                    for (int i = 0; i < (int)g_emissiveTextures.size(); ++i) {
                        bool sel = (o.emissiveIndex == i);
                        if (ImGui::Selectable(g_emissiveTextures[i].name.c_str(), sel)) o.emissiveIndex = i;
                    }
                    ImGui::EndCombo();
                }

                ImGui::Checkbox("Nontrivial Sampling", &o.useNontrivialSampling);
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
        updateModelUniforms();
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
            2, 1, &vk_texture_set, 0, nullptr);

        auto drawObject = [&](int index, const ObjectState& o) {
            PushConstants pc;
            pc.albedoIndex = o.albedoIndex;
            pc.specularIndex = o.specularIndex;
            pc.emissiveIndex = o.emissiveIndex;
            pc.useNontrivialSampling = o.useNontrivialSampling ? 1 : 0;
            pc.time = (float)ImGui::GetTime();
            vkCmdPushConstants(cmd, vk_pipeline_layout,
                VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                0, sizeof(PushConstants), &pc);

            uint32_t dynamicOffset = (uint32_t)(index * g_model_ubo_stride);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, vk_pipeline_layout,
                1, 1, &vk_model_set, 1, &dynamicOffset);

            vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);
            };

        drawObject(0, obj1);
        drawObject(1, obj2);

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace application