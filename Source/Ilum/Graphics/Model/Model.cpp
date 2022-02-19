#include "Model.hpp"

#include <Graphics/Device/Device.hpp>

#include "Graphics/Command/CommandBuffer.hpp"
#include "Graphics/GraphicsContext.hpp"



#include <Core/JobSystem/JobSystem.hpp>

namespace Ilum
{
Model::Model(Model &&other) noexcept :
    submeshes(std::move(other.submeshes)),
    bounding_box(other.bounding_box),
    vertices_count(other.vertices_count),
    indices_count(other.indices_count),
    mesh(std::move(other.mesh)),
    meshlets(std::move(other.meshlets))
{
	other.submeshes.clear();
}

Model &Model::operator=(Model &&other) noexcept
{
	submeshes      = std::move(other.submeshes);
	bounding_box   = other.bounding_box;
	vertices_count = other.vertices_count;
	indices_count  = other.indices_count;
	mesh           = std::move(other.mesh);
	meshlets       = std::move(other.meshlets);

	other.submeshes.clear();

	return *this;
}
}        // namespace Ilum