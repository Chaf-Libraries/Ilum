#pragma once

#include "Graphics/Model/Model.hpp"

#include "Material/Material.h"

namespace Ilum::cmpt
{
struct Renderable
{
	Renderable()
	{
		update = true;
	}

	~Renderable()
	{
		update = true;
	}

	inline static bool update = false;
};

// Meshlet Renderer only for static external mesh
struct MeshletRenderer : public Renderable
{
	std::string model;

	std::vector<scope<Material>> materials;
};

enum class MeshType
{
	None,
	Model,
	Sphere,
	Plane,
};

// Dynamic Mesh use MeshRenderer, set as default texture
struct MeshRenderer : public Renderable
{
	MeshType type = MeshType::None;

	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	Buffer vertex_buffer;
	Buffer index_buffer;

	uint32_t material_id = 0;
	bool     need_update = true;

	scope<Material> material = createScope<material::PBRMaterial>();
};
}        // namespace Ilum::cmpt
