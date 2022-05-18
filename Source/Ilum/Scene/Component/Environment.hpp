#pragma once

#include "Component.hpp"

#include <Asset/AssetManager.hpp>

#include <entt.hpp>

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::cmpt
{
enum class EnvironmentType
{
	Skybox,
	Procedure
};

class Environment : public Component
{
  public:
	Environment() = default;

	virtual void Tick(Scene &scene, entt::entity entity, RHIDevice *device) override;

	virtual bool OnImGui(ImGuiContext &context) override;

	Texture *GetCubemap();

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(m_type);
	}

  private:
	EnvironmentType m_type = EnvironmentType::Skybox;

	Texture *m_skybox = nullptr;

	AssetManager *m_manager = nullptr;

	std::unique_ptr<Texture> m_cubemap;
};
}        // namespace Ilum::cmpt