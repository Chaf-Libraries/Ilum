#pragma once

#include "RenderCore/MaterialGraph/MaterialNode.hpp"

namespace Ilum::MGNode
{
template <typename T>
struct BxDFNode : public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t &handle)
	{
		MaterialNodeDesc desc;
		return desc.SetName<T>();
	}

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo& info) = 0;
};

STRUCT(BlendBxDFNode, Enable, MaterialNode("Blend BxDF"), Category("BxDF")) :
    public MaterialNode
{
	STRUCT(BlendWeight, Enable)
	{
		META(Editor("Slider"), Min(0.f), Max(1.f), Name(""))
		float weight = 0.5f;
	};

	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo& info) override;
};
}        // namespace Ilum::MGNode