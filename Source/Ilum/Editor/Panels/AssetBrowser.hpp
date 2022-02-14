#pragma once

#include "Editor/Panel.hpp"

#include <Graphics/Resource/Image.hpp>

namespace Ilum::panel
{
class AssetBrowser : public Panel
{
  public:
	AssetBrowser();

	~AssetBrowser() = default;

	virtual void draw(float delta_time) override;

  private:
	Graphics::Image m_model_icon;
	Graphics::Image m_shader_icon;
};
}        // namespace Ilum::panel