#pragma once

#include "Editor/Panel.hpp"

#include "Graphics/Image/Image.hpp"

namespace Ilum::panel
{
class AssetBrowser : public Panel
{
  public:
	AssetBrowser();

	~AssetBrowser() = default;

	virtual void draw(float delta_time) override;

  private:
	Image m_model_icon;
	Image m_shader_icon;
};
}        // namespace Ilum::panel