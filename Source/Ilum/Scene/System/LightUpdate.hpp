#pragma once

#include "Utils/PCH.hpp"

#include "Scene/System.hpp"

namespace Ilum::sym
{
class LightUpdate : public System
{
  public:
	LightUpdate() = default;

	~LightUpdate() = default;

	virtual void run() override;
};
}        // namespace Ilum::sym