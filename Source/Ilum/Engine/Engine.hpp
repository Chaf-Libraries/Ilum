#pragma once

#include <Core/Time.hpp>

namespace Ilum
{
class Window;
class RHIContext;
class Editor;

class Engine
{
  public:
	Engine();

	~Engine();

	void Tick();

  private:
	std::unique_ptr<Window>     m_window      = nullptr;
	std::unique_ptr<RHIContext> m_rhi_context = nullptr;
	std::unique_ptr<Editor> m_editor = nullptr;

	Timer m_timer;
};
}        // namespace Ilum