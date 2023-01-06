#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct MaterialData
{
	std::unordered_map<std::string, RHISampler *> samplers;
	std::unordered_map<std::string, RHITexture *> textures;

	std::string shader = "Asset/Material/default.material.hlsli";

	void Bind(RHIPipelineState* pipeline_state)
	{

	}

	void Bind(RHIDescriptor *descriptor)
	{
		for (auto &[name, sampler] : samplers)
		{
			descriptor->BindSampler(name, sampler);
		}

		for (auto &[name, texture] : textures)
		{
			descriptor->BindTexture(name, texture, RHITextureDimension::Texture2D);
		}
	}
};
}        // namespace Ilum