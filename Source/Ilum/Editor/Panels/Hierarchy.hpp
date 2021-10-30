#pragma once

#include "Editor/Panel.hpp"

namespace Ilum::panel
{
	class Hierarchy :public Panel
	{
	  public:
	    Hierarchy();

		virtual ~Hierarchy() = default;

		virtual void draw(float delta_time) override;
	};
}