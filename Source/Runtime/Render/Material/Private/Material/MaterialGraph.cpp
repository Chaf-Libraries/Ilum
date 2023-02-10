#include "MaterialGraph.hpp"

namespace Ilum
{
MaterialGraphDesc &MaterialGraphDesc::SetName(const std::string &name)
{
	m_name = name;
	return *this;
}

MaterialGraphDesc &MaterialGraphDesc::AddNode(size_t handle, MaterialNodeDesc &&desc)
{
	for (auto &[pin_handle, pin] : desc.GetPins())
	{
		m_node_lookup[pin_handle] = handle;
	}

	m_node_lookup[handle] = handle;
	desc.SetHandle(handle);
	m_nodes.emplace(handle, std::move(desc));
	return *this;
}

void MaterialGraphDesc::EraseNode(size_t handle)
{
	const auto &desc = m_nodes.at(handle);
	const auto &pins = desc.GetPins();

	for (auto iter = m_edges.begin(); iter != m_edges.end();)
	{
		if (pins.find(iter->first) != pins.end() ||
		    pins.find(iter->second) != pins.end())
		{
			iter = m_edges.erase(iter);
		}
		else
		{
			iter++;
		}
	}

	for (auto &[handle, name] : pins)
	{
		m_node_lookup.erase(handle);
	}

	m_nodes.erase(handle);
}

void MaterialGraphDesc::EraseLink(size_t source, size_t target)
{
	if (m_edges.find(target) != m_edges.end() &&
	    m_edges.at(target) == source)
	{
		m_edges.erase(target);
	}
}

MaterialGraphDesc &MaterialGraphDesc::Link(size_t source, size_t target)
{
	const auto &src_node = m_nodes.at(m_node_lookup.at(source));
	const auto &dst_node = m_nodes.at(m_node_lookup.at(target));

	const auto &src_pin = src_node.GetPin(source);
	const auto &dst_pin = dst_node.GetPin(target);

	if ((src_pin.type & dst_pin.accept) &&
	    src_pin.attribute != dst_pin.attribute)
	{
		m_edges[target] = source;
	}

	return *this;
}

bool MaterialGraphDesc::HasLink(size_t target) const
{
	return m_edges.find(target) != m_edges.end();
}

size_t MaterialGraphDesc::LinkFrom(size_t target) const
{
	return m_edges.at(target);
}

const MaterialNodeDesc &MaterialGraphDesc::GetNode(size_t handle) const
{
	return m_nodes.at(m_node_lookup.at(handle));
}

const std::string &MaterialGraphDesc::GetName() const
{
	return m_name;
}

const std::map<size_t, MaterialNodeDesc> &MaterialGraphDesc::GetNodes() const
{
	return m_nodes;
}

const std::map<size_t, size_t> &MaterialGraphDesc::GetEdges() const
{
	return m_edges;
}

void MaterialGraphDesc::Clear()
{
	m_nodes.clear();
	m_edges.clear();
	m_node_lookup.clear();
}

}        // namespace Ilum