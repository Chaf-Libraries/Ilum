#include "IMaterialNode.hpp"

using namespace Ilum;

class SRGBToLinear : public MaterialNode<SRGBToLinear>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("SRGBToLinear")
		    .SetCategory("Converter")
		    .Input(handle++, "In", MaterialNodePin::Type::RGB, MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(0.f))
		    .Output(handle++, "Out", MaterialNodePin::Type::Float3);
	}

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, ResourceManager *manager, MaterialCompilationContext *context) override
	{
		if (context->IsCompiled(node_desc))
		{
			return;
		}

		std::map<std::string, std::string> parameters;
		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("In"), graph_desc, manager, context);
		context->variables.emplace_back(fmt::format("float3 S_{} = SRGBtoLINEAR({});", node_desc.GetPin("Out").handle, parameters["In"]));
	}
};

CONFIGURATION_MATERIAL_NODE(SRGBToLinear)