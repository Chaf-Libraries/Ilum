#include "SemanticNode.hpp"
#include "MaterialGraph/MaterialGraph.hpp"

namespace Ilum::MGNode
{
MaterialNodeDesc OutputNode::Create(size_t &handle)
{
	MaterialNodeDesc desc;
	return desc.SetName<OutputNode>()
	    .AddPin(handle, "Surface BSDF", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input)
	    .AddPin(handle, "Volumetric BSDF", MaterialNodePin::Type::BSDF, MaterialNodePin::Attribute::Input);
}

bool OutputNode::Validate(const MaterialNodeDesc &node, MaterialGraphDesc &graph)
{
	return graph.HasLink(node.GetPin("Surface BSDF").handle);
}

void OutputNode::EmitHLSL(const MaterialNodeDesc &desc, MaterialGraphDesc &graph, MaterialEmitInfo &info)
{
	size_t surface_bsdf_pin = graph.LinkFrom(desc.GetPin("Surface BSDF").handle);

	const auto &surface_bsdf_node_desc = graph.GetNode(surface_bsdf_pin);

	auto surface_bsdf_node = rttr::type::get_by_name(surface_bsdf_node_desc.name).create();

	MaterialEmitInfo surface_bsdf_info;

	surface_bsdf_node.get_type().get_method("EmitHLSL").invoke(surface_bsdf_node, surface_bsdf_node_desc, graph, surface_bsdf_info).to_string();

	info.includes.insert(std::make_move_iterator(surface_bsdf_info.includes.begin()), std::make_move_iterator(surface_bsdf_info.includes.end()));
	info.definitions.insert(info.definitions.end(), std::make_move_iterator(surface_bsdf_info.definitions.begin()), std::make_move_iterator(surface_bsdf_info.definitions.end()));
	info.type_name = surface_bsdf_info.type_name;
}
}        // namespace Ilum::MGNode