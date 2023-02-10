#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct MaterialData
{
	std::vector<uint32_t> textures;
	std::vector<uint32_t> samplers;

	std::string shader    = "Material/Material.hlsli";
	std::string signature = "Signature_0";

	void Reset()
	{
		textures.clear();
		samplers.clear();
		shader    = "Material/Material.hlsli";
		signature = "Signature_0";
	}

	template<typename Archive>
	void serialize(Archive& archive)
	{
		archive(textures, samplers, shader, signature);
	}
};
}        // namespace Ilum