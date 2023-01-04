#include "IMaterialNode.hpp"

using namespace Ilum;

class VectorMerge : public MaterialNode<VectorMerge>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("VectorMerge")
		    .SetCategory("Converter")
		    .Input(handle++, "X", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "Y", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "Z", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Output(handle++, "Out", MaterialNodePin::Type::Float3);
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
		context->SetParameter<float>(parameters, node_desc.GetPin("X"), graph_desc, renderer, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Y"), graph_desc, renderer, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Z"), graph_desc, renderer, context);
		context->variables.emplace_back(fmt::format("float3 S_{} = float3({}, {}, {});", node_desc.GetPin("Out").handle, parameters["X"], parameters["Y"], parameters["Z"]));
	}
};

CONFIGURATION_MATERIAL_NODE(VectorMerge)