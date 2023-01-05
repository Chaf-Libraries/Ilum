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

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor* editor) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, ResourceManager *manager, MaterialCompilationContext *context) override
	{
		if (context->IsCompiled(node_desc))
		{
			return;
		}

		context->variables.emplace_back(fmt::format("float S_{} = instance_attribute.instance_id;", node_desc.GetPin("InstanceID").handle));
		context->variables.emplace_back(fmt::format("float S_{} = instance_attribute.material_id;", node_desc.GetPin("MaterialID").handle));
		context->variables.emplace_back(fmt::format("float S_{} = instance_attribute.primitive_id;", node_desc.GetPin("PrimitiveID").handle));
	}
};

CONFIGURATION_MATERIAL_NODE(InstanceAttribute)