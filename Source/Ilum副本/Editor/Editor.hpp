#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Subsystem.hpp"

#include "Panel.hpp"

#include "Scene/Entity.hpp"

namespace Ilum
{
class Editor : public TSubsystem<Editor>
{
  public:
	Editor(Context *context);

	~Editor() = default;

	virtual bool onInitialize() override;

	virtual void onPreTick() override;

	virtual void onTick(float delta_time) override;

	virtual void onPostTick() override;

	virtual void onShutdown() override;

	void select(Entity entity);

	Entity getSelect();

  private:
	std::vector<scope<Panel>> m_panels;

	Entity m_select_entity;

	std::string m_scene_path = "";
};
}        // namespace Ilum