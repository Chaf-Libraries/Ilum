#pragma once

#include "RenderCore/MaterialGraph/MaterialNode.hpp"

namespace Ilum::MGNode
{
STRUCT(RGB, Enable, MaterialNode("RGB"), Category("Input")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void Update(MaterialNodeDesc & node) override;

	virtual void Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context) override;

	virtual void EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context) override;

	STRUCT(Data, Enable)
	{
		META(Editor("ColorEdit"), Name(""))
		glm::vec3 color = glm::vec3(0.f);
	};
};

STRUCT(ObjectInfo, Enable, MaterialNode("ObjectInfo"), Category("Input")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void Update(MaterialNodeDesc & node) override;

	virtual void Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context) override;

	virtual void EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context) override;
};
}        // namespace Ilum::MGNode