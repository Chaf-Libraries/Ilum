#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct MaterialData
{
	std::unordered_map<std::string, RHISampler *> samplers;
	std::unordered_map<std::string, RHITexture *> textures;

	std::string shader = "Material/Material.hlsli";
	std::string signature = "Signature_0";

	void Bind(RHIPipelineState *pipeline_state) const
	{

	}

	void Bind(RHIDescriptor *descriptor) const
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