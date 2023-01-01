#include "IMaterialNode.hpp"

using namespace Ilum;

class InstanceAttribute : public MaterialNode<InstanceAttribute>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("InstanceAttribute")
		    .SetCategory("Input")
		    .Output(handle++, "InstanceID", MaterialNodePin::Type::Float)
		    .Output(handle++, "MaterialID", MaterialNodePin::Type::Float)
		    .Output(handle++, "PrimitiveID", MaterialNodePin::Type::Float);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, MaterialGraph *graph) override
	{
	}
};

CONFIGURATION_MATERIAL_NODE(InstanceAttribute)