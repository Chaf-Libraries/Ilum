#pragma once

#include <cstdint>

namespace Ilum::Graphics
{
struct VideoMode
{
	uint32_t width        = 0;
	uint32_t height       = 0;
	uint32_t red_bits     = 0;
	uint32_t green_bits   = 0;
	uint32_t blue_bits    = 0;
	uint32_t refresh_rate = 0;
};

class Monitor
{
  public:
	Monitor(const VideoMode &video_mode);
	~Monitor() = default;

	uint32_t GetWidth() const;
	uint32_t GetHeight() const;
	uint32_t GetRedBits() const;
	uint32_t GetGreenBits() const;
	uint32_t GetBlueBits() const;
	uint32_t GetRefreshRate() const;

  private:
	VideoMode m_video_mode;
};
}        // namespace Ilum