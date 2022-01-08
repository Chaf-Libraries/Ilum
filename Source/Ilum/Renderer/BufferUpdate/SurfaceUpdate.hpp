#pragma once

#include "Utils/PCH.hpp"

#include "Scene/System.hpp"

namespace Ilum::sym
{
class SurfaceUpdate : public System
{
  public:
	SurfaceUpdate() = default;

	~SurfaceUpdate() = default;

	virtual void run() override;
};
}        // namespace Ilum::sym