#pragma once

#include "RenderCore/MaterialGraph/MaterialNode.hpp"

#include <RHI/RHISampler.hpp>

namespace Ilum::MGNode
{
STRUCT(ExternalTexture, Enable, MaterialNode("ExternalTexture"), Category("Resource")) :
    public MaterialNode
{
	STRUCT(TextureID, Enable)
	{
		META(Editor("Texture"), DragDrop("Texture"))
		size_t id = ~0;
	};

	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo& info) override;
};

STRUCT(SamplerState, Enable, MaterialNode("SamplerState"), Category("Resource")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo& info) override;
};
}        // namespace Ilum::MGNode