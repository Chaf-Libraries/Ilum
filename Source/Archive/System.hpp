#pragma once

#include <Renderer/Renderer.hpp>
#include <Resource/ResourceManager.hpp>
#include <Scene/Component/AllComponent.hpp>
#include <Scene/Entity.hpp>
#include <Scene/Scene.hpp>

namespace Ilum
{
class System
{
  public:
	static System &GetInstance();

	template <typename... Tn>
	void Execute(Renderer *renderer)
	{
	}

	void Tick(Renderer *renderer);
};
}        // namespace Ilum

#include "System.inl"