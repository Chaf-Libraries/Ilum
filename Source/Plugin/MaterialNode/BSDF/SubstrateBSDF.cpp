#include "IMaterialNode.hpp"

using namespace Ilum;

class SubstrateBSDF : public MaterialNode<SubstrateBSDF>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("SubstrateBSDF")
		    .SetCategory("BSDF")
		    .Input(handle++, "Rd", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
		    .Input(handle++, "Rs", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
		    .Input(handle++, "Roughness", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float, float(0.5f))
		    .Input(handle++, "Anisotropic", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float, float(0.f))
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

		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("Rd"), graph_desc, manager, context);
		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("Rs"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Roughness"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Anisotropic"), graph_desc, manager, context);

		context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
		    fmt::format("S_{}", node_desc.GetPin("Out").handle),
		    "SubstrateBSDF",
		    fmt::format("S_{}.Init({}, {}, {}, {}, {});", node_desc.GetPin("Out").handle, parameters["Rd"], parameters["Rs"], parameters["Roughness"], parameters["Anisotropic"], parameters["Normal"])});
	}
};

CONFIGURATION_MATERIAL_NODE(SubstrateBSDF)