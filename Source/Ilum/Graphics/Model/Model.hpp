#pragma once

#include "SubMesh.hpp"

#include "Graphics/Buffer/Buffer.h"

namespace Ilum
{
class Model
{
  public:
	Model() = default;

	Model(std::vector<SubMesh> &&submeshes);

	~Model() = default;

	Model(const Model &) = delete;

	Model &operator=(const Model &) = delete;

	Model(Model &&other) noexcept;

	Model &operator=(Model &&other) noexcept;

	const std::vector<SubMesh> &getSubMeshes() const;

  private:
	void createBuffer();

  private:
	std::vector<SubMesh> m_submeshes;

	Buffer m_vertex_buffer;
	Buffer m_index_buffer;
};

using ModelReference = std::reference_wrapper<const Model>;
}        // namespace Ilum