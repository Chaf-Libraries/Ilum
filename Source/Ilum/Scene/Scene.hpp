#pragma once

#include <RHI/Device.hpp>
#include <RHI/ImGuiContext.hpp>

#include <entt.hpp>

namespace Ilum
{
class AssetManager;

struct GeometryBatch
{
	std::vector<Buffer *> vertex_buffers;
	std::vector<Buffer *> index_buffers;
	std::vector<Buffer *> meshlet_vertex_buffers;
	std::vector<Buffer *> meshlet_triangle_buffers;
	std::vector<Buffer *> meshlet_buffers;
	std::vector<Buffer *> meshlet_bound_buffers;
	std::vector<AccelerationStructure *> bottom_level_acceleration_structures;
};

class Scene
{
  public:
	Scene(RHIDevice *device, AssetManager &asset_manager, const std::string &name);
	~Scene() = default;

	entt::registry &GetRegistry();

	AssetManager &GetAssetManager();

	const std::string &GetName() const;
	const std::string &GetSavePath() const;

	void Clear();

	entt::entity Create(const std::string &name = "Untitled Entity");

	void Tick();

	void OnImGui(ImGuiContext &context);

	void Save(const std::string &filename = "");
	void Load(const std::string &filename);

	void ImportGLTF(const std::string &filename);
	void ExportGLTF(const std::string &filename);

  private:
	RHIDevice *p_device = nullptr;

	AssetManager &m_asset_manager;

	entt::registry m_registry;

	entt::entity m_select = entt::null;

	std::string m_name      = "";
	std::string m_save_path = "";

  private:
	std::unique_ptr<Buffer> m_instance_buffer = nullptr;
	std::unique_ptr<Buffer> m_transform_buffer = nullptr;
	std::unique_ptr<AccelerationStructure> m_top_level_acceleration_structure = nullptr;

	bool m_transform_update = false;
	bool m_mesh_update      = false;
};
}        // namespace Ilum