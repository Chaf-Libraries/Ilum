#include "IMaterialNode.hpp"

using namespace Ilum;

class VectorSplit : public MaterialNode<VectorSplit>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("VectorSplit")
		    .SetCategory("Converter")
		    .Input(handle++, "In", MaterialNodePin::Type::Float3, MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(0.f))
		    .Output(handle++, "X", MaterialNodePin::Type::Float)
		    .Output(handle++, "Y", MaterialNodePin::Type::Float)
		    .Output(handle++, "Z", MaterialNodePin::Type::Float);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, MaterialGraph *graph) override
	{
	}
};

CONFIGURATION_MATERIAL_NODE(VectorSplit)