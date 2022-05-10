#pragma once

#include <RHI/AccelerateStructure.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/ImGuiContext.hpp>
#include <RHI/Texture.hpp>

#include <Render/ShaderInterop.hpp>

#include <Geometry/BoundingBox.hpp>

#include <glm/glm.hpp>

namespace Ilum
{
class AssetManager;
class Material;

class Submesh
{
	friend class Scene;

  public:
	Submesh(RHIDevice *device, AssetManager &manager);
	~Submesh() = default;

	const std::string &GetName() const;

	bool OnImGui(ImGuiContext &context);

	void UpdateBuffer();

	const BoundingBox &GetBoundingBox();

	Material *GetMaterial();

	Buffer &GetVertexBuffer();
	Buffer &GetIndexBuffer();
	Buffer &GetMeshletVertexBuffer();
	Buffer &GetMeshletTriangleBuffer();
	Buffer &GetMeshletBuffer();
	Buffer &GetMeshletBoundBuffer();

	AccelerationStructure &GetBLAS();

  private:
	RHIDevice *p_device = nullptr;

	AssetManager &m_manager;

	std::string m_name;

	BoundingBox m_bounding_box;

	std::vector<ShaderInterop::Vertex>       m_vertices;
	std::vector<uint32_t>                    m_indices;
	std::vector<uint32_t>                    m_meshlet_vertices;
	std::vector<uint32_t>                    m_meshlet_triangles;
	std::vector<ShaderInterop::Meshlet>      m_meshlets;
	std::vector<ShaderInterop::MeshletBound> m_meshlet_bounds;

	Material *m_material = nullptr;

	std::unique_ptr<Buffer> m_vertex_buffer           = nullptr;
	std::unique_ptr<Buffer> m_index_buffer            = nullptr;
	std::unique_ptr<Buffer> m_meshlet_vertex_buffer   = nullptr;
	std::unique_ptr<Buffer> m_meshlet_triangle_buffer = nullptr;
	std::unique_ptr<Buffer> m_meshlet_buffer          = nullptr;
	std::unique_ptr<Buffer> m_meshlet_bound_buffer    = nullptr;

	std::unique_ptr<AccelerationStructure> m_bottom_level_acceleration_structure = nullptr;
};

class Mesh
{
	friend class Scene;

  public:
	Mesh(RHIDevice *device);
	~Mesh() = default;

	const std::string &GetName() const;

	void UpdateBuffer();

	bool OnImGui(ImGuiContext &context);

	const std::vector<std::unique_ptr<Submesh>> &GetSubmeshes() const;

  private:
	std::string m_name;

	RHIDevice *p_device = nullptr;

	std::vector<std::unique_ptr<Submesh>> m_submeshes;
};
}        // namespace Ilum