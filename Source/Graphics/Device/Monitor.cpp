#include "Monitor.hpp"

#include <SDL.h>
#include <SDL_syswm.h>

namespace Ilum::Graphics
{
Monitor::Monitor(const VideoMode &video_mode) :
    m_video_mode(video_mode)
{
}

uint32_t Monitor::GetWidth() const
{
	return m_video_mode.width;
}

uint32_t Monitor::GetHeight() const
{
	return m_video_mode.height;
}

uint32_t Monitor::GetRedBits() const
{
	return m_video_mode.red_bits;
}

uint32_t Monitor::GetGreenBits() const
{
	return m_video_mode.green_bits;
}

uint32_t Monitor::GetBlueBits() const
{
	return m_video_mode.blue_bits;
}

uint32_t Monitor::GetRefreshRate() const
{
	return m_video_mode.refresh_rate;
}
}        // namespace Ilum