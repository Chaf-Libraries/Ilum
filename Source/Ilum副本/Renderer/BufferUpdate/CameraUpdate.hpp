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

  private:
	std::vector<glm::vec2> m_jitter_samples;
};
}        // namespace Ilum::sym