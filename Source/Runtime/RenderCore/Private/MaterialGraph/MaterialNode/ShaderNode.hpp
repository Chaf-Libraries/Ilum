#pragma once

#include "RenderCore/MaterialGraph/MaterialNode.hpp"

namespace Ilum::MGNode
{
STRUCT(AddShader, Enable, MaterialNode("Add Shader"), Category("Shader")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void Update(MaterialNodeDesc & node) override;

	virtual void Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context) override;

	virtual void EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context) override;
};

STRUCT(MixShader, Enable, MaterialNode("Mix Shader"), Category("Shader")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void Update(MaterialNodeDesc & node) override;

	virtual void Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context) override;

	virtual void EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context) override;

	STRUCT(MixShaderData, Enable)
	{
		META(Min(0.f), Max(1.f), Editor("Slider"), Name(""))
		float frac = 0.5f;
	};
};

STRUCT(DiffuseBSDF, Enable, MaterialNode("Diffuse BSDF"), Category("Shader")) :
    public MaterialNode
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void Update(MaterialNodeDesc & node) override;

	virtual void Validate(const MaterialNodeDesc &node, MaterialGraph *graph, ShaderValidateContext &context) override;

	virtual void EmitShader(const MaterialNodeDesc &desc, MaterialGraph *graph, ShaderEmitContext &context) override;

	STRUCT(DiffuseBSDFColor, Enable)
	{
		META(Editor("ColorEdit"), Name(""))
		glm::vec3 color = glm::vec3(1.f);
	};

	STRUCT(DiffuseBSDFRoughness, Enable)
	{
		META(Min(0.f), Max(1.f), Editor("Slider"), Name(""))
		float roughness = 0.5f;
	};
};

}        // namespace Ilum::MGNode