#pragma once

#include "MaterialGraph/MaterialGraph.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
void MaterialGraphDesc::AddNode(size_t &handle, MaterialNodeDesc &&desc)
{
	for (auto &[pin_handle, name] : desc.pin_index)
	{
		node_query[pin_handle] = handle;
		desc.handle            = handle;
	}
	nodes.emplace(handle++, std::move(desc));
}

void MaterialGraphDesc::EraseNode(size_t handle)
{
	const auto &desc = nodes.at(handle);

	for (auto iter = links.begin(); iter != links.end();)
	{
		if (desc.pin_index.find(iter->first) != desc.pin_index.end() ||
		    desc.pin_index.find(iter->second) != desc.pin_index.end())
		{
			iter = links.erase(iter);
		}
		else
		{
			iter++;
		}
	}

	for (auto &[handle, name] : desc.pin_index)
	{
		node_query.erase(handle);
	}

	nodes.erase(handle);
}

void MaterialGraphDesc::EraseLink(size_t source, size_t target)
{
	if (links.find(target) != links.end() &&
	    links.at(target) == source)
	{
		links.erase(target);
	}
}

void MaterialGraphDesc::Link(size_t source, size_t target)
{
	const auto &src_node = nodes.at(node_query.at(source));
	const auto &dst_node = nodes.at(node_query.at(target));

	const auto &src_pin = src_node.GetPin(source);
	const auto &dst_pin = dst_node.GetPin(target);

	if ((src_pin.type & dst_pin.type) &&
	    src_pin.attribute != dst_pin.attribute)
	{
		links[target] = source;
	}
}

bool MaterialGraphDesc::HasLink(size_t target)
{
	return links.find(target) != links.end();
}

size_t MaterialGraphDesc::LinkFrom(size_t target_pin)
{
	return links.at(target_pin);
}

const MaterialNodeDesc &MaterialGraphDesc::GetNode(size_t pin)
{
	return nodes.at(node_query.at(pin));
}

std::string MaterialGraphDesc::GetEmitResult(const MaterialNodeDesc &desc, const std::string &pin_name, MaterialEmitInfo &emit_info)
{
	if (HasLink(desc.GetPin(pin_name).handle))
	{
		size_t source_link = LinkFrom(desc.GetPin(pin_name).handle);

		const auto &link_node_desc = GetNode(LinkFrom(desc.GetPin(pin_name).handle));
		auto        link_node      = rttr::type::get_by_name(link_node_desc.name).create();
		link_node.get_type().get_method("EmitHLSL").invoke(link_node, link_node_desc, *this, emit_info);
		if (emit_info.IsExpression(LinkFrom(desc.GetPin(pin_name).handle)))
		{
			return emit_info.expression.at(LinkFrom(desc.GetPin(pin_name).handle));
		}
		else
		{
			return "S" + std::to_string(LinkFrom(desc.GetPin(pin_name).handle));
		}
	}
	else
	{
		return "";
	}
}

std::string MaterialGraphDesc::GetEmitExpression(const MaterialNodeDesc &desc, const std::string &pin_name, MaterialEmitInfo &emit_info)
{
	if (HasLink(desc.GetPin(pin_name).handle))
	{
		size_t source_pin = LinkFrom(desc.GetPin(pin_name).handle);

		if (emit_info.expression.find(source_pin) != emit_info.expression.end())
		{
			return emit_info.expression.at(source_pin);
		}

		const auto &source_node_desc = GetNode(source_pin);
		auto        source_node        = rttr::type::get_by_name(source_node_desc.name).create();
		source_node.get_type().get_method("EmitHLSL").invoke(source_node, source_node_desc, *this, emit_info);

		return emit_info.expression.at(source_pin);
	}
	else
	{
		return "";
	}
}

MaterialGraph::MaterialGraph(RHIContext *rhi_context) :
    p_rhi_context(rhi_context)
{
}

}        // namespace Ilum