#pragma once

#include "Editor/Panel.hpp"

namespace Ilum::panel
{
class MainCameraSetting : public Panel
{
  public:
	MainCameraSetting();

	~MainCameraSetting() = default;

	virtual void draw(float delta_time) override;
};
}        // namespace Ilum::panel