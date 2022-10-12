#include "VariableNode.hpp"
#include "MaterialGraph/MaterialGraph.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc RGB::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc
	    .AddPin(handle, "RGB", MaterialNodePin::Type::Float3, MaterialNodePin::Attribute::Input, RGBColor())
	    .AddPin(handle, "Out", MaterialNodePin::Type::Float3, MaterialNodePin::Attribute::Output)
	    .SetName<RGB>();

	return desc;
}

void RGB::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{
	if (graph.HasLink(desc.GetPin("RGB").handle))
	{
		const auto &variable_desc = graph.GetNode(graph.LinkFrom(desc.GetPin("RGB").handle));
		auto        variable_node = rttr::type::get_by_name(variable_desc.name).create();
		variable_node.get_type().get_method("EmitHLSL").invoke(variable_node, variable_desc, graph, info);
		info.definitions.push_back(fmt::format("float3 S{} = {};", desc.GetPin("Out").handle, "S" + std::to_string(graph.LinkFrom(desc.GetPin("RGB").handle))));
	}
	else
	{
		RGBColor color = desc.GetPin("RGB").data.convert<RGBColor>();
		info.definitions.push_back(fmt::format("float3 S{} = float3({}, {}, {});", desc.GetPin("Out").handle, color.color.x, color.color.y, color.color.z));
	}
}

MaterialNodeDesc RGBA::Create(size_t &handle)
{
	MaterialNodeDesc desc;

	desc
	    .AddPin(handle, "RGBA", MaterialNodePin::Type::Float4, MaterialNodePin::Attribute::Input, RGBAColor())
	    .AddPin(handle, "Out", MaterialNodePin::Type::Float4, MaterialNodePin::Attribute::Output)
	    .SetName<RGBA>();

	return desc;
}

void RGBA::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{
	if (graph.HasLink(desc.GetPin("RGB").handle))
	{
		const auto &variable_desc = graph.GetNode(graph.LinkFrom(desc.GetPin("RGBA").handle));
		auto        variable_node = rttr::type::get_by_name(variable_desc.name).create();
		variable_node.get_type().get_method("EmitHLSL").invoke(variable_node, variable_desc, graph, info);
		info.definitions.push_back(fmt::format("float4 S{} = {};", desc.GetPin("Out").handle, "S" + std::to_string(graph.LinkFrom(desc.GetPin("RGBA").handle))));
	}
	else
	{
		RGBAColor color = desc.GetPin("RGBA").data.convert<RGBAColor>();
		info.definitions.push_back(fmt::format("float4 S{} = float4({}, {}, {});", desc.GetPin("Out").handle, color.color.x, color.color.y, color.color.z, color.color.w));
	}
}
}        // namespace Ilum::MGNode