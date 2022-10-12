#include "ResourceNode.hpp"
#include "MaterialGraph/MaterialGraph.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc ExternalTexture::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc.AddPin(handle, "Out", MaterialNodePin::Type::Texture2D, MaterialNodePin::Attribute::Output)
	    .SetData(TextureID())
	    .SetName<ExternalTexture>();

	return desc;
}

void ExternalTexture::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{

}

MaterialNodeDesc SamplerState::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc
	    .AddPin(handle, "In", MaterialNodePin::Type::Texture2D | MaterialNodePin::Type::Texture3D | MaterialNodePin::Type::TextureCube, MaterialNodePin::Attribute::Input)
	    .SetData(SamplerStateType{})
	    .AddPin(handle, "Out",
	            MaterialNodePin::Type::Float | MaterialNodePin::Type::Int | MaterialNodePin::Type::Uint |
	                MaterialNodePin::Type::Float2 | MaterialNodePin::Type::Int2 | MaterialNodePin::Type::Uint2 |
	                MaterialNodePin::Type::Float3 | MaterialNodePin::Type::Int3 | MaterialNodePin::Type::Uint3 |
	                MaterialNodePin::Type::Float4 | MaterialNodePin::Type::Int4 | MaterialNodePin::Type::Uint4,
	            MaterialNodePin::Attribute::Output)
	    .SetName<SamplerState>();

	return desc;
}

void SamplerState::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{
	if (graph.HasLink(desc.GetPin("In").handle))
	{
		const auto &variable_desc = graph.GetNode(graph.LinkFrom(desc.GetPin("In").handle));
		auto        variable_node = rttr::type::get_by_name(variable_desc.name).create();
		variable_node.get_type().get_method("EmitHLSL").invoke(variable_node, variable_desc, graph, info);

		if (info.IsExpression(graph.LinkFrom(desc.GetPin("In").handle)))
		{
			//info.expression.emplace(desc.GetPin("Out"), fmt::format("TextureArray[{}].Sample"));
		}
	}
}
}        // namespace Ilum::MGNode