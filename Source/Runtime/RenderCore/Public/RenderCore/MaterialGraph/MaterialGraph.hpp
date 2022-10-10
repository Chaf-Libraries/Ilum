#pragma once

#include "MaterialNode.hpp"

namespace Ilum
{
STRUCT(MaterialGraphDesc, Enable)
{
	std::map<size_t, MaterialNodeDesc> nodes;

	std::map<size_t, size_t> links;        // Target - Source

	std::map<size_t, size_t> node_query;

	void AddNode(size_t & handle, MaterialNodeDesc && desc);

	void EraseNode(size_t handle);

	void EraseLink(size_t source, size_t target);

	void Link(size_t source, size_t target);

	bool HasLink(size_t target);
};
}        // namespace Ilum