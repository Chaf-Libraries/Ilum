#pragma once

#include <RHI/Device.hpp>
#include <RHI/ImGuiContext.hpp>

#include "Component/Camera.hpp"
#include "Component/Hierarchy.hpp"
#include "Component/MeshRenderer.hpp"
#include "Component/Tag.hpp"
#include "Component/Transform.hpp"

#include <Asset/Material.hpp>

#include <entt.hpp>

namespace Ilum
{
class AssetManager;

struct GeometryBatch
{
	std::vector<cmpt::MeshRenderer *> meshes;
	std::vector<uint32_t>             order;
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

	AccelerationStructure &GetTLAS();
	Buffer                &GetMainCameraBuffer();

	entt::entity GetMainCamera();

	void Clear();

	entt::entity Create(const std::string &name = "Untitled Entity");

	void Tick();

	void OnImGui(ImGuiContext &context);

	void Save(const std::string &filename = "");
	void Load(const std::string &filename);

	void ImportGLTF(const std::string &filename);
	void ExportGLTF(const std::string &filename);

	GeometryBatch Batch(AlphaMode mode);

  private:
	RHIDevice *p_device = nullptr;

	AssetManager &m_asset_manager;

	entt::registry m_registry;

	entt::entity m_select      = entt::null;
	entt::entity m_main_camera = entt::null;

	std::string m_name      = "";
	std::string m_save_path = "";

  private:
	std::unique_ptr<AccelerationStructure> m_top_level_acceleration_structure = nullptr;

	std::unique_ptr<Buffer> m_main_camera_buffer = nullptr;

	uint32_t m_instance_count = 0;
	uint32_t m_meshlet_count  = 0;

	bool m_update = false;
};
}        // namespace Ilum