#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Subsystem.hpp"

#include "EntityManager.hpp"
#include "SystemManager.hpp"

namespace Ilum
{
class Entity;

class Scene : public TSubsystem<Scene>
{
  public:
	Scene(Context *context = nullptr);

	~Scene() = default;

	void load(const std::string &filepath);

	void clear();


	

  private:
	std::string m_name = "untitled_scene";

	EntityManager m_entity_manager;
	SystemManager m_system_manager;
};
}        // namespace Ilum