#pragma once

#include "Editor/Panel.hpp"

namespace Ilum::panel
{
class MeshModifier : public Panel
{
  public:
	MeshModifier();

	virtual ~MeshModifier() = default;

	virtual void draw(float delta_time) override;

  public:
	bool active = true;
};
}        // namespace Ilum::panel