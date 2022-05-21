#pragma once

#include <RHI/Device.hpp>
#include <RHI/ImGuiContext.hpp>

#include "Component/MeshRenderer.hpp"

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

	std::vector<Buffer *> GetInstanceBuffer();

	entt::entity GetMainCamera();

	entt::entity GetSelected();

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
	void UpdateTransform();
	void UpdateTransformRecursive(entt::entity entity);
	void UpdateCamera();
	void UpdateInstance();
	void UpdateTLAS();
	void UpdateLights();
	void UpdateEnvironment();

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

	bool m_update = false;
};
}        // namespace Ilum