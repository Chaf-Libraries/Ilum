#include "Model.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Synchronization/Queue.hpp"
#include "Graphics/Synchronization/QueueSystem.hpp"

#include "Threading/ThreadPool.hpp"

namespace Ilum
{
Model::Model(Model &&other) noexcept :
    submeshes(std::move(other.submeshes)),
    bounding_box(other.bounding_box),
    vertices_count(other.vertices_count),
    indices_count(other.indices_count),
    meshlet_count(other.meshlet_count),
    vertices_offset(other.vertices_offset),
    indices_offset(other.indices_offset),
    meshlet_offset(other.meshlet_offset),
    meshlets_buffer(std::move(meshlets_buffer)),
    vertices_buffer(std::move(vertices_buffer)),
    indices_buffer(std::move(indices_buffer))
{
	other.submeshes.clear();
}

Model &Model::operator=(Model &&other) noexcept
{
	submeshes       = std::move(other.submeshes);
	bounding_box    = other.bounding_box;
	vertices_count  = other.vertices_count;
	indices_count   = other.indices_count;
	meshlet_count   = other.meshlet_count;
	vertices_offset = other.vertices_offset;
	indices_offset  = other.indices_offset;
	meshlet_offset  = other.meshlet_offset;
	meshlets_buffer  = std::move(meshlets_buffer);
	vertices_buffer = std::move(vertices_buffer);
	indices_buffer  = std::move(indices_buffer);

	other.submeshes.clear();

	return *this;
}
}        // namespace Ilum