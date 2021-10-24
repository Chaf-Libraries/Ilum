#pragma once

#include "Editor/Panel.hpp"

namespace Ilum::panel
{
class Inspector : public Panel
{
  public:
	Inspector();

	~Inspector() = default;

	virtual void draw() override;
};
}        // namespace Ilum::panel