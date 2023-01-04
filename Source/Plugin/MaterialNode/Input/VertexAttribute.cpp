#include "IMaterialNode.hpp"

using namespace Ilum;

class VertexAttribute : public MaterialNode<VertexAttribute>
{
  public:
	virtual MaterialNodeDesc Create(size_t &handle) override
	{
		MaterialNodeDesc desc;
		return desc
		    .SetHandle(handle++)
		    .SetName("VertexAttribute")
		    .SetCategory("Input")
		    .Output(handle++, "Position", MaterialNodePin::Type::Float3)
		    .Output(handle++, "Normal", MaterialNodePin::Type::Float3)
		    .Output(handle++, "Tangent", MaterialNodePin::Type::Float3)
		    .Output(handle++, "Texcoord0", MaterialNodePin::Type::Float3)
		    .Output(handle++, "Texcoord1", MaterialNodePin::Type::Float3);
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

		context->variables.emplace_back(fmt::format("float3 S_{} = vertex_attribute.position;", node_desc.GetPin("Position").handle));
		context->variables.emplace_back(fmt::format("float3 S_{} = vertex_attribute.normal;", node_desc.GetPin("Normal").handle));
		context->variables.emplace_back(fmt::format("float3 S_{} = vertex_attribute.tangent;", node_desc.GetPin("Tangent").handle));
		context->variables.emplace_back(fmt::format("float3 S_{} = vertex_attribute.texcoord0;", node_desc.GetPin("Texcoord0").handle));
		context->variables.emplace_back(fmt::format("float3 S_{} = vertex_attribute.texcoord1;", node_desc.GetPin("Texcoord1").handle));
	}
};

CONFIGURATION_MATERIAL_NODE(VertexAttribute)