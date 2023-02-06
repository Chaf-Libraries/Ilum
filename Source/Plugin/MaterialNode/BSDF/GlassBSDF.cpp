#include "IMaterialNode.hpp"

using namespace Ilum;

class GlassBSDF : public MaterialNode<GlassBSDF>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("GlassBSDF")
		    .SetCategory("BSDF")
		    .Input(handle++, "Reflectance", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
		    .Input(handle++, "Transmission", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
		    .Input(handle++, "Roughness", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.5f))
		    .Input(handle++, "Refraction", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(1.5f))
		    .Input(handle++, "Anisotropic", MaterialNodePin::Type::Float, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, float(0.f))
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

		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("Reflectance"), graph_desc, manager, context);
		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("Transmission"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Roughness"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Refraction"), graph_desc, manager, context);
		context->SetParameter<float>(parameters, node_desc.GetPin("Anisotropic"), graph_desc, manager, context);

		context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
		    fmt::format("S_{}", node_desc.GetPin("Out").handle),
		    "GlassBSDF",
		    fmt::format("S_{}.Init({}, {}, {}, {}, {}, {});", node_desc.GetPin("Out").handle,
		                parameters["Reflectance"],
		                parameters["Transmission"],
		                parameters["Roughness"],
		                parameters["Refraction"],
		                parameters["Anisotropic"],
		                parameters["Normal"])});
	}
};

CONFIGURATION_MATERIAL_NODE(GlassBSDF)