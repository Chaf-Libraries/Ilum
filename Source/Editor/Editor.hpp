#pragma once

#include "Panel.hpp"

#include <memory>
#include <vector>

namespace Ilum::Editor
{
class Editor
{
  public:
	Editor();
	~Editor() = default;

	void Show();

  public:
	void ShowPanels();
	void ShowMenu();

  private:
	std::vector<std::unique_ptr<Panel>> m_panels;
};
}        // namespace Ilum::Editor