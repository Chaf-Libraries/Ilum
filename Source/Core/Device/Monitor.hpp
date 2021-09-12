#pragma once

#include "Core/Engine/PCH.hpp"

//#include "SDL.h"

namespace Ilum
{
struct VideoMode
{
	uint32_t width = 0;
	uint32_t height = 0;
	uint32_t red_bits = 0;
	uint32_t green_bits = 0;
	uint32_t blue_bits  = 0;
	uint32_t refresh_rate = 0;
};

class Monitor
{
  public:
	Monitor(const VideoMode &video_mode);

	~Monitor() = default;

	uint32_t getWidth() const;

	uint32_t getHeight() const;

	uint32_t getRedBits() const;

	uint32_t getGreenBits() const;

	uint32_t getBlueBits() const;

	uint32_t getRefreshRate() const;

  private:
	VideoMode m_video_mode;
};
}        // namespace Ilum