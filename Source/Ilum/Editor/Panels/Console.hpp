#pragma once

#include "Editor/Panel.hpp"

#include "Graphics/Image/Image.hpp"

#include <imgui.h>

namespace Ilum::panel
{
class Console : public Panel
{
  public:
	Console();

	~Console() = default;

	virtual void draw(float delta_time) override;

  private:
	ImGuiTextFilter m_filter;

	Image m_icons[6];
};
}        // namespace Ilum::panel