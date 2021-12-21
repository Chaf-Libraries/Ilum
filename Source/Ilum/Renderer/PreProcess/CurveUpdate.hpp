#pragma once

#include "Utils/PCH.hpp"

#include "Scene/System.hpp"

namespace Ilum::sym
{
class CurveUpdate : public System
{
  public:
	CurveUpdate() = default;

	~CurveUpdate() = default;

	virtual void run() override;
};
}        // namespace Ilum::sym