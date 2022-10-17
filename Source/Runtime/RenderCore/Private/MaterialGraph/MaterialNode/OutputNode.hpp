#pragma once

#include "RenderCore/MaterialGraph/MaterialNode.hpp"

namespace Ilum::MGNode
{
STRUCT(MaterialOutput, Enable, MaterialNode("Material Output"), Category("Output")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void Update(MaterialNodeDesc & node) override;

	virtual void Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context) override;

	virtual void EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context) override;
};
}