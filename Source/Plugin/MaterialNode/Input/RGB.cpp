#include "IMaterialNode.hpp"

using namespace Ilum;

class RGB : public MaterialNode<RGB>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("RGB")
		    .SetCategory("Input")
		    .Output(handle++, "Color", MaterialNodePin::Type::RGB, glm::vec3(0.f));
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, MaterialGraph *graph, MaterialCompilationContext &context) override
	{
	}
};

CONFIGURATION_MATERIAL_NODE(RGB)