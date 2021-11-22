#include "SubMesh.hpp"

#include "Graphics/Command/CommandBuffer.hpp"

namespace Ilum
{
SubMesh::SubMesh(std::vector<Vertex> &&vertices, std::vector<uint32_t> &&indices, uint32_t index_offset, scope<material::DisneyPBR> &&material) :
    vertices(std::move(vertices)),
    indices(std::move(indices)),
    index_offset(index_offset),
    material(*material)
{
	for (auto &vertex : this->vertices)
	{
		bounding_box.merge(vertex.position);
	}

	indirect_cmd.firstIndex    = this->index_offset;
	indirect_cmd.firstInstance = 0;
	indirect_cmd.indexCount    = static_cast<uint32_t>(this->indices.size());
	indirect_cmd.instanceCount = 1;
	indirect_cmd.vertexOffset  = this->vertex_offset;
}

SubMesh::SubMesh(SubMesh &&other) noexcept :
    vertices(std::move(other.vertices)),
    indices(std::move(other.indices)),
    index_offset(other.index_offset),
    bounding_box(other.bounding_box),
    material(std::move(other.material)),
    indirect_cmd(other.indirect_cmd)
{
	other.vertices.clear();
	other.indices.clear();
	other.index_offset = 0;
}

SubMesh &SubMesh::operator=(SubMesh &&other) noexcept
{
	vertices     = std::move(other.vertices);
	indices      = std::move(other.indices);
	index_offset = other.index_offset;
	bounding_box = other.bounding_box;
	material     = std::move(other.material);
	indirect_cmd = other.indirect_cmd;

	other.vertices.clear();
	other.indices.clear();
	other.index_offset = 0;

	return *this;
}
}        // namespace Ilum