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
	std::array<glm::vec2, 16> m_halton_sequence;
	uint32_t                  m_frame_count = 0;
};
}        // namespace Ilum::sym