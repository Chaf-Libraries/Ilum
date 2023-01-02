#include "IMaterialNode.hpp"

using namespace Ilum;

class PrincipledBSDF : public MaterialNode<PrincipledBSDF>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("PrincipledBSDF")
		    .SetCategory("BSDF")
		    .Input(handle++, "Color", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
		    .Input(handle++, "Roughness", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Output(handle++, "Out", MaterialNodePin::Type::BSDF);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, MaterialGraph *graph, MaterialCompilationContext &context) override
	{
	}
};

CONFIGURATION_MATERIAL_NODE(PrincipledBSDF)