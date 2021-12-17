#pragma once

#include "Utils/PCH.hpp"

#include "Scene/System.hpp"

namespace Ilum::sym
{
class CameraUpdate : public System
{
  public:
	CameraUpdate() = default;

	~CameraUpdate() = default;

	virtual void run() override;
};
}        // namespace Ilum::sym