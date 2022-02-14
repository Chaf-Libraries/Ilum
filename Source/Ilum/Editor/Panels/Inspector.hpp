#pragma once

#include "Editor/Panel.hpp"

#include <Graphics/Resource/Image.hpp>

namespace Ilum::panel
{
class Inspector : public Panel
{
  public:
	Inspector();

	virtual ~Inspector() = default;

	virtual void draw(float delta_time) override;
};
}        // namespace Ilum::panel