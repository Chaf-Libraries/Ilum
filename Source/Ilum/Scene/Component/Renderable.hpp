#pragma once

#include "Graphics/Model/Model.hpp"

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
struct StaticMeshRenderer : public Renderable
{
	std::string model;

	std::vector<Material> materials;
};

enum class MeshType
{
	None,
	Model,
	Sphere,
	Plane,
};

// Dynamic Mesh use DynamicMeshRenderer, set as default texture
struct DynamicMeshRenderer : public Renderable
{
	MeshType type = MeshType::None;

	std::vector<Vertex>   vertices;
	std::vector<uint32_t> indices;

	Buffer vertex_buffer;
	Buffer index_buffer;

	uint32_t material_id = 0;
	bool     need_update = true;

	Material material;
};

enum class CurveType
{
	None,
	BezierCurve,
	BSplineCurve,
	CubicSplineCurve,
	RationalBezierCurve,
	RationalBSplineCurve,
};

struct CurveRenderer : public Renderable
{
	CurveType type = CurveType::None;

	std::vector<glm::vec3> control_points;

	std::vector<glm::vec3> vertices;

	std::vector<float> weights;

	uint32_t order = 4;

	Buffer vertex_buffer;

	bool need_update = true;

	// Curve only support base color
	glm::vec4 base_color = glm::vec4(1.f);

	float line_width = 1.f;

	uint32_t sample = 100;

	uint32_t select_point = std::numeric_limits<uint32_t>::max();
};

enum class SurfaceType
{
	None,
	BezierSurface,
	BSplineSurface,
	RationalBezierSurface,
	RationalBSplineSurface,
};

struct SurfaceRenderer : public Renderable
{
	SurfaceType type = SurfaceType::None;

	std::vector<std::vector<glm::vec3>> control_points;

	std::vector<Vertex> vertices;

	std::vector<uint32_t> indices;

	std::vector<std::vector<float>> weights;

	uint32_t order = 4;

	Buffer vertex_buffer;

	Buffer index_buffer;

	bool need_update = true;

	uint32_t sample_x = 20;

	uint32_t sample_y = 20;

	uint32_t select_point[2] = {std::numeric_limits<uint32_t>::max(), std::numeric_limits<uint32_t>::max()};
};
}        // namespace Ilum::cmpt
