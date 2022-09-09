#pragma once

#include "Widget.hpp"

namespace Ilum
{
class ResourceBrowser : public Widget
{
  public:
	ResourceBrowser(Editor *editor);

	~ResourceBrowser();

	virtual void Tick() override;

  private:
	void DrawTextureBrowser();
	void DrawSceneBrowser();
	void DrawRenderGraphBrowser();

  private:
	const float m_button_size = 70.f;
};
}        // namespace Ilum