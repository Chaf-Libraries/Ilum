#pragma once

#include "Utils/PCH.hpp"

#include "Scene/System.hpp"

#include <glm/glm.hpp>

namespace Ilum::sym
{
class CameraUpdate : public System
{
  public:
	CameraUpdate();

	~CameraUpdate() = default;

	virtual void run() override;
};
}        // namespace Ilum::sym