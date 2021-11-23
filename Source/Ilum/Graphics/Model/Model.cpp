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
    vertices(std::move(other.vertices)),
    indices(std::move(other.indices))
{
	other.submeshes.clear();
}

Model &Model::operator=(Model &&other) noexcept
{
	submeshes      = std::move(other.submeshes);
	bounding_box   = other.bounding_box;
	vertices_count = other.vertices_count;
	indices_count  = other.indices_count;
	vertices       = std::move(other.vertices);
	indices       = std::move(other.indices);

	other.submeshes.clear();

	return *this;
}
}        // namespace Ilum