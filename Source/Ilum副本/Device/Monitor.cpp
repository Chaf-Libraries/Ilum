#include "Monitor.hpp"

#include <SDL.h>
#include <SDL_syswm.h>

namespace Ilum
{
Monitor::Monitor(const VideoMode &video_mode):
    m_video_mode(video_mode)
{
}

uint32_t Monitor::getWidth() const
{
	return m_video_mode.width;
}

uint32_t Monitor::getHeight() const
{
	return m_video_mode.height;
}

uint32_t Monitor::getRedBits() const
{
	return m_video_mode.red_bits;
}

uint32_t Monitor::getGreenBits() const
{
	return m_video_mode.green_bits;
}

uint32_t Monitor::getBlueBits() const
{
	return m_video_mode.blue_bits;
}

uint32_t Monitor::getRefreshRate() const
{
	return m_video_mode.refresh_rate;
}
}        // namespace Ilum