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

	static Engine* instance();

	void onTick();

	Context &getContext();

  private:
	scope<Context> m_context = nullptr;

	static Engine *s_instance;
};
}        // namespace Ilum