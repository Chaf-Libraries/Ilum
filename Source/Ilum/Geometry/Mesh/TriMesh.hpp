#pragma once

#include <vector>

#include "Geometry/Vertex.hpp"

namespace Ilum::geometry
{
struct TriMesh
{
	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	TriMesh() = default;

	TriMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices);

	TriMesh(TriMesh &&other) noexcept;

	TriMesh &operator=(TriMesh &&other) noexcept;

	~TriMesh() = default;

	void clear();

	bool empty() const;
};
}        // namespace Ilum::geometry