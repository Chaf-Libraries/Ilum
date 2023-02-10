#include "IMaterialNode.hpp"

using namespace Ilum;

class ThinDielectricMaterial : public MaterialNode<ThinDielectricMaterial>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("ThinDielectricMaterial")
		    .SetCategory("BSDF")
		    .Input(handle++, "Reflectance", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
		    .Input(handle++, "Transmittance", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
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

		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("Reflectance"), graph_desc, manager, context);
		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("Transmittance"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("IOR"), graph_desc, manager, context);

		context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
		    fmt::format("S_{}", node_desc.GetPin("Out").handle),
		    "ThinDielectricMaterial",
		    fmt::format("S_{}.Init({}, {}, {}, {});", node_desc.GetPin("Out").handle,
		                parameters["Reflectance"],
		                parameters["Transmittance"],
		                parameters["IOR"],
		                parameters["Normal"])});
	}
};

CONFIGURATION_MATERIAL_NODE(ThinDielectricMaterial)