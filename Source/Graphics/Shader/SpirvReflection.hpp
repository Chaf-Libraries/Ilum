#pragma once

#include "../Vulkan.hpp"

#include <unordered_set>

namespace Ilum::Graphics
{
enum class ShaderResourceQualifiers : uint16_t
{
	None         = 0,
	NonReadable  = 1,
	NonWriteable = 2
};

inline ShaderResourceQualifiers operator|(ShaderResourceQualifiers lhs, ShaderResourceQualifiers rhs)
{
	return static_cast<ShaderResourceQualifiers>(static_cast<uint16_t>(lhs) | static_cast<uint16_t>(rhs));
}

inline ShaderResourceQualifiers operator&(ShaderResourceQualifiers lhs, ShaderResourceQualifiers rhs)
{
	return static_cast<ShaderResourceQualifiers>(static_cast<uint16_t>(lhs) & static_cast<uint16_t>(rhs));
}

struct ReflectionData
{
	struct Attribute
	{
		enum class Type
		{
			None,
			Input,
			Output
		};

		std::string           name       = "";
		uint32_t              location   = 0;
		uint32_t              vec_size   = 0;
		uint32_t              array_size = 0;
		uint32_t              columns    = 0;
		Type                  type       = Type::None;
		VkShaderStageFlagBits stage      = VK_SHADER_STAGE_ALL;
	};

	struct InputAttachment
	{
		std::string           name                   = "";
		uint32_t              array_size             = 0;
		uint32_t              input_attachment_index = 0;
		uint32_t              set                    = 0;
		uint32_t              binding                = 0;
		bool                  bindless               = false;
		VkShaderStageFlagBits stage                  = VK_SHADER_STAGE_FRAGMENT_BIT;
	};

	struct Image
	{
		enum class Type
		{
			None,
			ImageSampler,
			Image,
			ImageStorage,
			Sampler
		};

		std::string              name       = "";
		uint32_t                 array_size = 0;        // When array_size = 0, enable descriptor indexing
		uint32_t                 set        = 0;
		uint32_t                 binding    = 0;
		bool                     bindless   = false;
		VkShaderStageFlags       stage      = 0;
		Type                     type       = Type::None;
		ShaderResourceQualifiers qualifiers = ShaderResourceQualifiers::None;        // For storage image
	};

	struct Buffer
	{
		enum class Type
		{
			None,
			Uniform,
			Storage
		};

		std::string              name       = "";
		uint32_t                 size       = 0;
		uint32_t                 array_size = 0;        // When array_size = 0, enable descriptor indexing
		uint32_t                 set        = 0;
		uint32_t                 binding    = 0;
		bool                     bindless   = false;
		VkShaderStageFlags       stage      = 0;
		Type                     type       = Type::None;
		ShaderResourceQualifiers qualifiers = ShaderResourceQualifiers::None;        // For storage buffer
	};

	struct Constant
	{
		enum class Type
		{
			None,
			Push,
			Specialization
		};

		std::string           name   = "";
		uint32_t              size   = 0;
		uint32_t              offset = 0;
		VkShaderStageFlagBits stage  = VK_SHADER_STAGE_ALL;
		Type                  type   = Type::None;

		// Only for Specialization constant
		uint32_t constant_id = 0;
	};

	VkShaderStageFlags           stage;
	std::vector<Attribute>       attributes;
	std::vector<InputAttachment> input_attachments;
	std::vector<Image>           images;
	std::vector<Buffer>          buffers;
	std::vector<Constant>        constants;
	std::unordered_set<uint32_t> sets;

	size_t hash = 0;

	ReflectionData &operator+=(const ReflectionData &rhs);

	void UpdateHash();
};

class SpirvReflection
{
  public:
	static ReflectionData Reflect(const std::vector<uint32_t> &spirv, VkShaderStageFlagBits stage);
};
}        // namespace Ilum::Graphics