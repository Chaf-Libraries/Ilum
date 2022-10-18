#pragma once

#include "MaterialGraph/MaterialGraph.hpp"

#include <RHI/RHIContext.hpp>

namespace Ilum
{
void MaterialGraphDesc::AddNode(size_t &handle, MaterialNodeDesc &&desc)
{
	for (auto &[pin_handle, pin] : desc.pins)
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
		if (desc.pins.find(iter->first) != desc.pins.end() ||
		    desc.pins.find(iter->second) != desc.pins.end())
		{
			iter = links.erase(iter);
		}
		else
		{
			iter++;
		}
	}

	for (auto &[handle, name] : desc.pins)
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

void MaterialGraphDesc::UpdateNode(size_t node_handle)
{
	auto &node_desc = nodes[node_handle];

	auto node = rttr::type::get_by_name(node_desc.name).create();

	node.get_type().get_method("Update").invoke(node, node_desc);
}

bool MaterialGraphDesc::HasLink(size_t target) const
{
	return links.find(target) != links.end();
}

size_t MaterialGraphDesc::LinkFrom(size_t target_pin) const
{
	return links.at(target_pin);
}

const MaterialNodeDesc &MaterialGraphDesc::GetNode(size_t pin) const
{
	return nodes.at(node_query.at(pin));
}

MaterialGraph::MaterialGraph(RHIContext *rhi_context, const MaterialGraphDesc &desc) :
    p_rhi_context(rhi_context), m_desc(desc)
{
}

MaterialGraphDesc &MaterialGraph::GetDesc()
{
	return m_desc;
}

void MaterialGraph::EmitShader(size_t pin, ShaderEmitContext &context)
{
	const auto &desc = m_desc.GetNode(pin);
	auto        node = rttr::type::get_by_name(desc.name).create();
	node.get_type().get_method("EmitShader").invoke(node, desc, this, context);
}

void MaterialGraph::Validate(size_t pin, ShaderValidateContext &context)
{
	const auto &desc = m_desc.GetNode(pin);
	auto        node = rttr::type::get_by_name(desc.name).create();
	node.get_type().get_method("Validate").invoke(node, desc, this, context);
}

void MaterialGraph::SetUpdate(bool update)
{
	m_update = update;
}

bool MaterialGraph::Update()
{
	return m_update;
}
}        // namespace Ilum