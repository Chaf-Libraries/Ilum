#pragma once

#include <RHI/RHIContext.hpp>

namespace Ilum
{
struct MaterialData
{
	std::vector<uint32_t> textures;
	std::vector<uint32_t> samplers;

	std::unique_ptr<RHIBuffer> uniform_buffer = nullptr;

	std::string shader    = "Material/Material.hlsli";
	std::string signature = "Signature_0";

	void Reset()
	{
		textures.clear();
		samplers.clear();
		shader    = "Material/Material.hlsli";
		signature = "Signature_0";
	}

	void Bind(RHIPipelineState *pipeline_state) const
	{
	}

	void Bind(RHIDescriptor *descriptor, const std::vector<RHITexture *> &textures, const std::vector<RHISampler *> &samplers) const
	{
		if (!textures.empty() || !samplers.empty())
		{
			descriptor
			    ->BindTexture("Textures", textures, RHITextureDimension::Texture2D)
			    .BindSampler("Samplers", samplers)
			    .BindBuffer("UniformBuffer", uniform_buffer.get());
		}
	}
};
}        // namespace Ilum