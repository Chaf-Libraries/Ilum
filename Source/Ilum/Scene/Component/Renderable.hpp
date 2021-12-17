#pragma once

#include "Graphics/Model/Model.hpp"

#include "Material/Material.h"

namespace Ilum::cmpt
{
struct Renderable
{
	inline static bool update = false;
};

// Meshlet Renderer only for static external mesh
struct MeshletRenderer : public Renderable
{
	std::string model;

	std::vector<scope<IMaterial>> materials;
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

	scope<IMaterial> material = createScope<material::DisneyPBR>();
};
}        // namespace Ilum::cmpt
