#include "Model.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Synchronization/Queue.hpp"
#include "Graphics/Synchronization/QueueSystem.hpp"

#include "Threading/ThreadPool.hpp"

namespace Ilum
{
Model::Model(std::vector<SubMesh> &&submeshes) :
    submeshes(std::move(submeshes))
{
	for (auto &submesh : this->submeshes)
	{
		bounding_box.merge(submesh.bounding_box);
	}

	for (auto &submesh : this->submeshes)
	{
		vertices_count += static_cast<uint32_t>(submesh.vertices.size());
		indices_count += static_cast<uint32_t>(submesh.indices.size());
	}
}

Model::Model(Model &&other) noexcept :
    submeshes(std::move(other.submeshes)),
    bounding_box(other.bounding_box),
    vertices_count(other.vertices_count),
    indices_count(other.indices_count)
{
	other.submeshes.clear();
}

Model &Model::operator=(Model &&other) noexcept
{
	submeshes     = std::move(other.submeshes);
	bounding_box  = other.bounding_box;
	vertices_count = other.vertices_count;
	indices_count  = other.indices_count;

	other.submeshes.clear();

	return *this;
}
}        // namespace Ilum