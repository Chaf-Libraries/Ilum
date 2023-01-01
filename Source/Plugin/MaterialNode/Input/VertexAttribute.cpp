#include "IMaterialNode.hpp"

using namespace Ilum;

class VertexAttribute : public MaterialNode<VertexAttribute>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("VertexAttribute")
		    .SetCategory("Input")
		    .Output(handle++, "Position", MaterialNodePin::Type::Float3)
		    .Output(handle++, "Normal", MaterialNodePin::Type::Float3)
		    .Output(handle++, "UVW0", MaterialNodePin::Type::Float3)
		    .Output(handle++, "UVW1", MaterialNodePin::Type::Float3);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, MaterialGraph *graph) override
	{
	}
};

CONFIGURATION_MATERIAL_NODE(VertexAttribute)