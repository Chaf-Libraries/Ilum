#pragma once

#include "../BxDFNode.hpp"

namespace Ilum::MGNode
{
STRUCT(LambertianReflection, Enable, MaterialNode("Lambertian Reflection"), Category("BxDF")) :
    public BxDFNode<LambertianReflection>
{
	virtual MaterialNodeDesc Create(size_t & handle) override;

	virtual void EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo& info) override;

	STRUCT(BaseColor, Enable){
		META(Editor("ColorEdit"), Name(""))
		glm::vec3 base_color = glm::vec3{0.f};
	};
};
}        // namespace Ilum::MGNode