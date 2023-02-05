#include "IMaterialNode.hpp"

using namespace Ilum;

class DisneyBSDF : public MaterialNode<DisneyBSDF>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("DisneyBSDF")
		    .SetCategory("BSDF")
		    .Input(handle++, "Normal", MaterialNodePin::Type::Float3, MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3)
		    .Input(handle++, "Reflectance", MaterialNodePin::Type::RGB, MaterialNodePin::Type::Float | MaterialNodePin::Type::RGB | MaterialNodePin::Type::Float3, glm::vec3(1.f))
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

		context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
		    fmt::format("S_{}", node_desc.GetPin("Out").handle),
		    "DisneyBSDF",
		    fmt::format("S_{}.Init({}, {});", node_desc.GetPin("Out").handle, parameters["Reflectance"], parameters["Normal"])});
	}
};

CONFIGURATION_MATERIAL_NODE(DisneyBSDF)