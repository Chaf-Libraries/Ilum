#pragma once

#include "RenderCore/MaterialGraph/MaterialNode.hpp"

namespace Ilum::MGNode
{
STRUCT(OutputNode, Enable, MaterialNode("Output"), Category("Semantic")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual bool Validate(const MaterialNodeDesc &node, MaterialGraphDesc &graph) override;

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo& info) override;
};
}