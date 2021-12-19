#pragma once

#include "Utils/PCH.hpp"

#include "Scene/System.hpp"

namespace Ilum::sym
{
class GeometryUpdate : public System
{
  public:
	GeometryUpdate() = default;

	~GeometryUpdate() = default;

	virtual void run() override;
};
}        // namespace Ilum::sym