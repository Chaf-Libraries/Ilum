#pragma once

#include "PCH.hpp"

namespace Ilum
{
class Context;

class Engine
{
  public:
	Engine();

	~Engine();

	void onTick();

	Context &getContext();

  private:
	scope<Context> m_context = nullptr;
};
}        // namespace Ilum