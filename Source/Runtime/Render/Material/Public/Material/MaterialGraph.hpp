#pragma once

#include "MaterialNode.hpp"

namespace Ilum
{
class EXPORT_API MaterialGraphDesc
{
  public:
	MaterialGraphDesc() = default;

	~MaterialGraphDesc() = default;

	MaterialGraphDesc &SetName(const std::string &name);

	MaterialGraphDesc &AddNode(size_t handle, MaterialNodeDesc &&desc);

	void EraseNode(size_t handle);

	void EraseLink(size_t source, size_t target);

	MaterialGraphDesc &Link(size_t source, size_t target);

	bool HasLink(size_t target) const;

	size_t LinkFrom(size_t target) const;

	const MaterialNodeDesc &GetNode(size_t handle) const;

	const std::string &GetName() const;

	const std::map<size_t, MaterialNodeDesc> &GetNodes() const;

	const std::map<size_t, size_t> &GetEdges() const;

	void Clear();

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(m_name, m_nodes, m_edges, m_node_lookup);
	}

  private:
	std::string m_name;

	std::map<size_t, MaterialNodeDesc> m_nodes;

	std::map<size_t, size_t> m_edges;        // Target - Source

	std::map<size_t, size_t> m_node_lookup;        // Pin ID - Node ID
};
}        // namespace Ilum