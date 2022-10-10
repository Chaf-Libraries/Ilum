#pragma once

#include "MaterialGraph/MaterialGraph.hpp"

namespace Ilum
{
void MaterialGraphDesc::AddNode(size_t &handle, MaterialNodeDesc &&desc)
{
	for (auto &[pin_handle, name] : desc.pin_index)
	{
		node_query[pin_handle] = handle;
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

	if (src_pin.type == dst_pin.type &&
	    src_pin.attribute != dst_pin.attribute)
	{
		links[target] = source;
	}
}

bool MaterialGraphDesc::HasLink(size_t target)
{
	return links.find(target) != links.end();
}
}        // namespace Ilum