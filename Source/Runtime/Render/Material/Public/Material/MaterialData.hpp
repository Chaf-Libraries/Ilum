#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
enum class BlendMode
{
	Opaque,
	Mask,
	Blend
};

struct MaterialData
{
	std::vector<uint32_t> textures;
	std::vector<uint32_t> samplers;

	std::string shader    = "Material/Material.hlsli";
	std::string signature = "Signature_0";

	BlendMode blend_mode = BlendMode::Opaque;

	void Reset()
	{
		textures.clear();
		samplers.clear();
		shader    = "Material/Material.hlsli";
		signature = "Signature_0";
	}

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(textures, samplers, shader, signature, blend_mode);
	}
};
}        // namespace Ilum