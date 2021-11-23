#include "SubMesh.hpp"

#include "Graphics/Command/CommandBuffer.hpp"

namespace Ilum
{
SubMesh::SubMesh(SubMesh &&other) noexcept :
    indices_offset(other.indices_offset),
    vertices_offset(other.vertices_offset),
    indices_count(other.indices_count),
    vertices_count(other.vertices_count),
    bounding_box(other.bounding_box),
    material(std::move(other.material)),
    pre_transform(std::move(other.pre_transform)),
    indirect_cmd(other.indirect_cmd)
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
	indirect_cmd = other.indirect_cmd;
	pre_transform   = other.pre_transform;

	return *this;
}
}        // namespace Ilum