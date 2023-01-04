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

	virtual void OnImGui(MaterialNodeDesc &node_desc, Editor *editor) override
	{
	}

	virtual void EmitHLSL(const MaterialNodeDesc &node_desc, const MaterialGraphDesc &graph_desc, Renderer* renderer, MaterialCompilationContext *context) override
	{
		if (context->IsCompiled(node_desc))
		{
			return;
		}

		std::map<std::string, std::string> parameters;
		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("In"), graph_desc, renderer, context);
		context->variables.emplace_back(fmt::format("float S_{} = {}.x;", node_desc.GetPin("X").handle, parameters["In"]));
		context->variables.emplace_back(fmt::format("float S_{} = {}.y;", node_desc.GetPin("Y").handle, parameters["In"]));
		context->variables.emplace_back(fmt::format("float S_{} = {}.z;", node_desc.GetPin("Z").handle, parameters["In"]));
	}
};

CONFIGURATION_MATERIAL_NODE(VectorSplit)