#pragma once

#include "SubMesh.hpp"

#include "Graphics/Buffer/Buffer.h"

#include "Geometry/BoundingBox.hpp"

namespace Ilum
{
class Model
{
  public:
	Model() = default;

	Model(std::vector<SubMesh> &&submeshes);

	~Model();

	Model(const Model &) = delete;

	Model &operator=(const Model &) = delete;

	Model(Model &&other) noexcept;

	Model &operator=(Model &&other) noexcept;

	const std::vector<SubMesh> &getSubMeshes() const;

	BufferReference getVertexBuffer() const;

	BufferReference getIndexBuffer() const;

	const geometry::BoundingBox &getBoundingBox() const;

  private:
	void createBuffer();

  private:
	std::vector<SubMesh> m_submeshes;

	Buffer m_vertex_buffer;
	Buffer m_index_buffer;

	geometry::BoundingBox m_bounding_box;
};

using ModelReference = std::reference_wrapper<const Model>;
}        // namespace Ilum