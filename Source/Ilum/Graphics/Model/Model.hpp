#pragma once

#include "SubMesh.hpp"

#include "Graphics/Buffer/Buffer.h"

#include "Geometry/BoundingBox.hpp"

namespace Ilum
{
struct Model
{
  public:
	std::vector<SubMesh> submeshes;

	Buffer vertex_buffer;
	Buffer index_buffer;

	geometry::BoundingBox bounding_box;

	Model() = default;

	Model(std::vector<SubMesh> &&submeshes);

	~Model() = default;

	Model(const Model &) = delete;

	Model &operator=(const Model &) = delete;

	Model(Model &&other) noexcept;

	Model &operator=(Model &&other) noexcept;

  private:
	void createBuffer();
};

using ModelReference = std::reference_wrapper<Model>;
}        // namespace Ilum