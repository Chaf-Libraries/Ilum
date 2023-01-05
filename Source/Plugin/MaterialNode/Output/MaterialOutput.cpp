#include "IMaterialNode.hpp"

using namespace Ilum;

class MaterialOutput : public MaterialNode<MaterialOutput>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("MaterialOutput")
		    .SetCategory("Output")
		    .Input(handle++, "Surface", MaterialNodePin::Type::BSDF)
		    .Input(handle++, "Volume", MaterialNodePin::Type::Media);
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

		auto &surface_bsdf_pin = node_desc.GetPin("Surface");
		auto &volume_bsdf_pin  = node_desc.GetPin("Volume");

		// Surface BSDF
		if (graph_desc.HasLink(surface_bsdf_pin.handle))
		{
			auto &surface_bsdf_node = graph_desc.GetNode(graph_desc.LinkFrom(surface_bsdf_pin.handle));
			surface_bsdf_node.EmitHLSL(graph_desc, manager, context);
		}

		// Volume BSDF
		if (graph_desc.HasLink(volume_bsdf_pin.handle))
		{
		}
	}
};

CONFIGURATION_MATERIAL_NODE(MaterialOutput)