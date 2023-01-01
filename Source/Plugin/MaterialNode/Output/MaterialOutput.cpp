#include "IMaterialNode.hpp"

using namespace Ilum;

class MaterialOutput : public MaterialNode<MaterialOutput>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("MaterialOutput")
		    .SetCategory("Output")
		    .Input(handle++, "Surface BSDF", MaterialNodePin::Type::BSDF)
		    .Input(handle++, "Volume BSDF", MaterialNodePin::Type::BSDF);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, MaterialGraph *graph) override
	{
	}
};

CONFIGURATION_MATERIAL_NODE(MaterialOutput)