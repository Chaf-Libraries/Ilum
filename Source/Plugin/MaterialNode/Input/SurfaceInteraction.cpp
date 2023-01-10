#include "IMaterialNode.hpp"

using namespace Ilum;

class SurfaceInteraction : public MaterialNode<SurfaceInteraction>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("SurfaceInteraction")
		    .SetCategory("Input")
		    .Output(handle++, "Position", MaterialNodePin::Type::Float3)
		    .Output(handle++, "Normal", MaterialNodePin::Type::Float3)
		    .Output(handle++, "UV", MaterialNodePin::Type::Float3)
		    .Output(handle++, "dUVdx", MaterialNodePin::Type::Float3)
		    .Output(handle++, "dUVdy", MaterialNodePin::Type::Float3);
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

		context->variables.emplace_back(fmt::format("float3 S_{} = surface_interaction.isect.p;", node_desc.GetPin("Position").handle));
		context->variables.emplace_back(fmt::format("float3 S_{} = surface_interaction.isect.n;", node_desc.GetPin("Normal").handle));
		context->variables.emplace_back(fmt::format("float3 S_{} = float3(surface_interaction.isect.uv, 0.f);", node_desc.GetPin("UV").handle));
		context->variables.emplace_back(fmt::format("float3 S_{} = float3(surface_interaction.duvdx, 0.f);", node_desc.GetPin("dUVdx").handle));
		context->variables.emplace_back(fmt::format("float3 S_{} = float3(surface_interaction.duvdy, 0.f);", node_desc.GetPin("dUVdy").handle));
	}
};

CONFIGURATION_MATERIAL_NODE(SurfaceInteraction)