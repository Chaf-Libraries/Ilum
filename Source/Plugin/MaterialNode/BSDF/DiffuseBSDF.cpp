#include "IMaterialNode.hpp"

using namespace Ilum;

class DiffuseBSDF : public MaterialNode<DiffuseBSDF>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("DiffuseBSDF")
		    .SetCategory("BSDF")
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

		context->SetParameter<glm::vec3>(parameters, node_desc.GetPin("Reflectance"), graph_desc, manager, context);

		context->bsdfs.emplace_back(MaterialCompilationContext::BSDF{
		    fmt::format("S_{}", node_desc.GetPin("Out").handle),
		    "DiffuseBSDF",
		    fmt::format("S_{}.Init({});", node_desc.GetPin("Out").handle, parameters["Reflectance"])});
	}
};

CONFIGURATION_MATERIAL_NODE(DiffuseBSDF)