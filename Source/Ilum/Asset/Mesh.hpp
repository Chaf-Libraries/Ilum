#pragma once

#include <RHI/AccelerateStructure.hpp>
#include <RHI/Buffer.hpp>
#include <RHI/ImGuiContext.hpp>
#include <RHI/Texture.hpp>

#include <Shaders/ShaderInterop.hpp>

#include <Geometry/BoundingBox.hpp>

#include <glm/glm.hpp>

namespace Ilum
{
class AssetManager;
class Material;

class Mesh
{
	friend class Scene;

  public:
	Mesh(RHIDevice *device, AssetManager &manager);
	~Mesh() = default;

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

	uint32_t GetVerticesCount() const;
	uint32_t GetIndicesCount() const;
	uint32_t GetMeshletVerticesCount() const;
	uint32_t GetMeshletTrianglesCount() const;
	uint32_t GetMeshletsCount() const;

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

	Material *m_material = nullptr;

	std::unique_ptr<Buffer> m_vertex_buffer           = nullptr;
	std::unique_ptr<Buffer> m_index_buffer            = nullptr;
	std::unique_ptr<Buffer> m_meshlet_vertex_buffer   = nullptr;
	std::unique_ptr<Buffer> m_meshlet_triangle_buffer = nullptr;
	std::unique_ptr<Buffer> m_meshlet_buffer          = nullptr;

	std::unique_ptr<AccelerationStructure> m_bottom_level_acceleration_structure = nullptr;
};
}        // namespace Ilum