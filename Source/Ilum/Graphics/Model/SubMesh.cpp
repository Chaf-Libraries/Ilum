#include "SubMesh.hpp"

#include "Graphics/Command/CommandBuffer.hpp"

namespace Ilum
{
SubMesh::SubMesh(SubMesh &&other) noexcept :
    indices_offset(other.indices_offset),
    vertices_offset(other.vertices_offset),
    indices_count(other.indices_count),
    vertices_count(other.vertices_count),
    meshlet_count(other.meshlet_count),
    meshlet_offset(other.meshlet_offset),
    bounding_box(other.bounding_box),
    material(std::move(other.material)),
    pre_transform(std::move(other.pre_transform))
{

}

SubMesh &SubMesh::operator=(SubMesh &&other) noexcept
{
	indices_offset = other.indices_offset;
	vertices_offset = other.vertices_offset;
	indices_count   = other.indices_count;
	vertices_count  = other.vertices_count;
	bounding_box = other.bounding_box;
	material     = std::move(other.material);
	pre_transform   = other.pre_transform;
	meshlet_offset  = other.meshlet_offset;
	meshlet_count   = other.meshlet_count;

	return *this;
}
}        // namespace Ilum