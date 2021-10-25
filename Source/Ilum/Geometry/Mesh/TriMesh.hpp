#pragma once

#include <vector>

namespace Ilum
{
template <typename Vertex>
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

	void set(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices);
};
}        // namespace Ilum

#include "TriMesh.inl"