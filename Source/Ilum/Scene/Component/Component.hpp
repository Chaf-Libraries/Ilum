#pragma once

#include <RHI/ImGuiContext.hpp>

#include "Serialize.hpp"

#include <entt.hpp>

namespace Ilum
{
class Scene;
}

namespace Ilum::cmpt
{
class Component
{
  public:
	Component()
	{
		m_update = true;
	}

	virtual void Tick(Scene &scene, entt::entity entity, RHIDevice *device)
	{}

	virtual bool OnImGui(ImGuiContext &context) = 0;

  protected:
	bool m_update = false;
};
}        // namespace Ilum::cmpt