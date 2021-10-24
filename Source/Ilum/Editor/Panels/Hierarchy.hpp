#pragma once

#include "Editor/Panel.hpp"

namespace Ilum::panel
{
	class Hierarchy :public Panel
	{
	  public:
	    Hierarchy();

		~Hierarchy() = default;

		virtual void draw() override;
	};
}