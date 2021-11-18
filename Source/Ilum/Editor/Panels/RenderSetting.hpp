#pragma once

#include "Utils/PCH.hpp"

#include "Editor/Panel.hpp"

namespace Ilum::panel
{
	class RenderSetting:public Panel
	{
	  public:
	    RenderSetting() = default;

		~RenderSetting() = default;

		virtual void draw(float delta_time) override;
    };
}