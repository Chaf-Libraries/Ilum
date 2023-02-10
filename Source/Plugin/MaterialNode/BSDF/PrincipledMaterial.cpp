#include "IMaterialNode.hpp"

using namespace Ilum;

class PrincipledMaterial : public MaterialNode<PrincipledMaterial>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("PrincipledMaterial")
		    .SetCategory("BSDF")
		    .Input(handle++, "BaseColor", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
		    .Input(handle++, "Metallic", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.5f))
		    .Input(handle++, "Roughness", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.5f))
		    .Input(handle++, "Anisotropic", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "Sheen", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "SheenTint", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "Specular", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "SpecTint", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "Clearcoat", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "ClearcoatGloss", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "Flatness", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "SpecTrans", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
		    .Input(handle++, "IOR", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(1.45f))
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
		else
		{
			parameters["Normal"] = fmt::format("ExtractNormalMap(surface_interaction, {})", parameters["Normal"]);
		}

		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("BaseColor"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Metallic"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Roughness"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Anisotropic"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Sheen"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("SheenTint"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Specular"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("SpecTint"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Clearcoat"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("ClearcoatGloss"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Flatness"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("SpecTrans"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("IOR"), graph_desc, manager, context);

		context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
		    fmt::format("S_{}", node_desc.GetPin("Out").handle),
		    "PrincipledMaterial",
		    fmt::format("S_{}.Init({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {});", node_desc.GetPin("Out").handle,
		                parameters["BaseColor"],
		                parameters["Metallic"],
		                parameters["Roughness"],
		                parameters["Anisotropic"],
		                parameters["Sheen"],
		                parameters["SheenTint"],
		                parameters["Specular"],
		                parameters["SpecTint"],
		                parameters["Clearcoat"],
		                parameters["ClearcoatGloss"],
		                parameters["Flatness"],
		                parameters["SpecTrans"],
		                parameters["IOR"],
		                parameters["Normal"])});
	}
};

CONFIGURATION_MATERIAL_NODE(PrincipledMaterial)