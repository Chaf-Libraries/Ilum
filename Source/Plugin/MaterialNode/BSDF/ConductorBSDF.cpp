#include "IMaterialNode.hpp"

using namespace Ilum;

class ConductorBSDF : public MaterialNode<ConductorBSDF>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("ConductorBSDF")
		    .SetCategory("BSDF")
		    .Input(handle++, "RoughnessU", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(1.f))
		    .Input(handle++, "RoughnessV", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(1.f))
		    .Input(handle++, "Eta", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(1.f))
		    .Input(handle++, "K", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(1.f))
		    .Input(handle++, "Normal", MaterialNodePin::Type::Float3, MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3)
		    .Output(handle++, "Out", MaterialNodePin::Type::BSDF);
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

		if (!context->HasParameter<glm::vec3>(parameters, node_desc.GetPin("Normal"), graph_desc, manager, context))
		{
			parameters["Normal"] = "surface_interaction.isect.n";
		}

		context->SetParameter<float>(parameters, node_desc.GetPin("RoughnessU"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("RoughnessV"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Eta"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("K"), graph_desc, manager, context);

		context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
		    fmt::format("S_{}", node_desc.GetPin("Out").handle),
		    "ConductorBSDF",
		    fmt::format("S_{}.Init({}, {}, {}, {}, {});", node_desc.GetPin("Out").handle,
		                parameters["RoughnessU"],
		                parameters["RoughnessV"],
		                parameters["Eta"],
		                parameters["K"],
		                parameters["Normal"])});
	}
};

CONFIGURATION_MATERIAL_NODE(ConductorBSDF)