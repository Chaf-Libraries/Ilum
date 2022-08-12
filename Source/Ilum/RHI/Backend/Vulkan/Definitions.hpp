#pragma once

#include "RHI/RHIDefinitions.hpp"

#include <volk.h>

#include <vk_mem_alloc.h>

#include <unordered_map>

namespace Ilum::Vulkan
{
inline static std::unordered_map<RHIFormat, VkFormat> ToVulkanFormat = {
    {RHIFormat::R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM},
    {RHIFormat::R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT},
    {RHIFormat::R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT},
    {RHIFormat::D32_SFLOAT, VK_FORMAT_D32_SFLOAT},
    {RHIFormat::D24_UNORM_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT},
};

inline static std::unordered_map<uint32_t, VkSampleCountFlagBits> ToVulkanSampleCountFlag = {
    {1, VK_SAMPLE_COUNT_1_BIT},
    {2, VK_SAMPLE_COUNT_2_BIT},
    {4, VK_SAMPLE_COUNT_4_BIT},
    {8, VK_SAMPLE_COUNT_8_BIT},
    {16, VK_SAMPLE_COUNT_16_BIT},
    {32, VK_SAMPLE_COUNT_32_BIT},
    {64, VK_SAMPLE_COUNT_64_BIT},
};

inline static std::unordered_map<RHITextureDimension, VkImageViewType> ToVulkanImageViewType = {
    {RHITextureDimension::Texture1D, VK_IMAGE_VIEW_TYPE_1D},
    {RHITextureDimension::Texture2D, VK_IMAGE_VIEW_TYPE_2D},
    {RHITextureDimension::Texture3D, VK_IMAGE_VIEW_TYPE_3D},
    {RHITextureDimension::TextureCube, VK_IMAGE_VIEW_TYPE_CUBE},
    {RHITextureDimension::Texture1DArray, VK_IMAGE_VIEW_TYPE_1D_ARRAY},
    {RHITextureDimension::Texture2DArray, VK_IMAGE_VIEW_TYPE_2D_ARRAY},
    {RHITextureDimension::TextureCubeArray, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY},
};

inline static std::unordered_map<RHIMemoryUsage, VmaMemoryUsage> ToVmaMemoryUsage = {
    {RHIMemoryUsage::GPU_Only, VMA_MEMORY_USAGE_GPU_ONLY},
    {RHIMemoryUsage::CPU_TO_GPU, VMA_MEMORY_USAGE_CPU_TO_GPU},
    {RHIMemoryUsage::GPU_TO_CPU, VMA_MEMORY_USAGE_GPU_TO_CPU},
};

inline static std::unordered_map<RHIFilter, VkFilter> ToVulkanFilter = {
    {RHIFilter::Linear, VK_FILTER_LINEAR},
    {RHIFilter::Nearest, VK_FILTER_NEAREST},
};

inline static std::unordered_map<RHIMipmapMode, VkSamplerMipmapMode> ToVulkanMipmapMode = {
    {RHIMipmapMode::Linear, VK_SAMPLER_MIPMAP_MODE_LINEAR},
    {RHIMipmapMode::Nearest, VK_SAMPLER_MIPMAP_MODE_NEAREST},
};

inline static std::unordered_map<RHIAddressMode, VkSamplerAddressMode> ToVulkanAddressMode = {
    {RHIAddressMode::Repeat, VK_SAMPLER_ADDRESS_MODE_REPEAT},
    {RHIAddressMode::Mirrored_Repeat, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT},
    {RHIAddressMode::Clamp_To_Edge, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE},
    {RHIAddressMode::Clamp_To_Border, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER},
    {RHIAddressMode::Mirror_Clamp_To_Edge, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE},
};

inline static std::unordered_map<RHISamplerBorderColor, VkBorderColor> ToVulkanBorderColor = {
    {RHISamplerBorderColor::Float_Transparent_Black, VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK},
    {RHISamplerBorderColor::Int_Transparent_Black, VK_BORDER_COLOR_INT_TRANSPARENT_BLACK},
    {RHISamplerBorderColor::Float_Opaque_Black, VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK},
    {RHISamplerBorderColor::Int_Opaque_Black, VK_BORDER_COLOR_INT_OPAQUE_BLACK},
    {RHISamplerBorderColor::Float_Opaque_White, VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE},
    {RHISamplerBorderColor::Int_Opaque_White, VK_BORDER_COLOR_INT_OPAQUE_WHITE},
};

inline static VkImageUsageFlags ToVulkanImageUsage(RHITextureUsage usage)
{
	VkImageUsageFlags vk_usage = 0;

	if (usage & RHITextureUsage::Transfer)
	{
		vk_usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
	}
	if (usage & RHITextureUsage::ShaderResource)
	{
		vk_usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
	}
	if (usage & RHITextureUsage::UnorderedAccess)
	{
		vk_usage |= VK_IMAGE_USAGE_STORAGE_BIT;
	}

	return vk_usage;
}

inline static VkBufferUsageFlags ToVulkanBufferUsage(RHIBufferUsage usage)
{
	VkBufferUsageFlags vk_usage = 0;

	if (usage & RHIBufferUsage::AccelerationStructure)
	{
		vk_usage |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
		            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
	}
	else if (usage & RHIBufferUsage::Index)
	{
		vk_usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
	}
	else if (usage & RHIBufferUsage::Vertex)
	{
		vk_usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
	}
	else if (usage & RHIBufferUsage::Indirect)
	{
		vk_usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
	}
	else if (usage & RHIBufferUsage::Transfer)
	{
		vk_usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
		            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	}
	else if (usage & RHIBufferUsage::ConstantBuffer)
	{
		vk_usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
	}
	else if (usage & RHIBufferUsage::UnorderedAccess)
	{
		vk_usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
		            VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
	}
	else if (usage & RHIBufferUsage::ShaderResource)
	{
		vk_usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
	}

	return vk_usage;
}

inline static VkShaderStageFlags ToVulkanShaderStage(RHIShaderStage stage)
{
	VkShaderStageFlags flag = 0;

	if (stage & RHIShaderStage::Vertex)
	{
		flag |= VK_SHADER_STAGE_VERTEX_BIT;
	}
	else if (stage & RHIShaderStage::Fragment)
	{
		flag |= VK_SHADER_STAGE_FRAGMENT_BIT;
	}
	else if (stage & RHIShaderStage::TessellationControl)
	{
		flag |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT;
	}
	else if (stage & RHIShaderStage::TessellationEvaluation)
	{
		flag |= VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
	}
	else if (stage & RHIShaderStage::Geometry)
	{
		flag |= VK_SHADER_STAGE_GEOMETRY_BIT;
	}
	else if (stage & RHIShaderStage::RayGen)
	{
		flag |= VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	}
	else if (stage & RHIShaderStage::AnyHit)
	{
		flag |= VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
	}
	else if (stage & RHIShaderStage::ClosestHit)
	{
		flag |= VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
	}
	else if (stage & RHIShaderStage::Miss)
	{
		flag |= VK_SHADER_STAGE_MISS_BIT_KHR;
	}
	else if (stage & RHIShaderStage::Intersection)
	{
		flag |= VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
	}
	else if (stage & RHIShaderStage::Callable)
	{
		flag |= VK_SHADER_STAGE_CALLABLE_BIT_KHR;
	}
	else if (stage & RHIShaderStage::Task)
	{
		flag |= VK_SHADER_STAGE_TASK_BIT_NV;
	}
	else if (stage & RHIShaderStage::Mesh)
	{
		flag |= VK_SHADER_STAGE_MESH_BIT_NV;
	}

	return flag;
}
}        // namespace Ilum::Vulkan