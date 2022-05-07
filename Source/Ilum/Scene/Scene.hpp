#pragma once

#include <RHI/Device.hpp>
#include <RHI/ImGuiContext.hpp>

#include <entt.hpp>

namespace Ilum
{
class Scene
{
  public:
	Scene(RHIDevice *device, const std::string &name);
	~Scene() = default;

	entt::registry &GetRegistry();

	void Clear();

	entt::entity Create(const std::string &name = "Untitled Entity");

	void Tick();

	void OnImGui(ImGuiContext &context);

	void Save(const std::string &filename);
	void Load(const std::string &filename);

  private:
	RHIDevice *p_device = nullptr;

	entt::registry m_registry;

	entt::entity m_select = entt::null;

	std::string m_name = "";

	bool m_update_transform = false;
};
}        // namespace Ilum