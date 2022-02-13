#include "Window.hpp"

#include <Core/Logger/Logger.hpp>

#include <SDL.h>
#include <SDL_syswm.h>

namespace Ilum::Graphics
{
Window::Window()
{
	// Initialize video subsystem
	if (SDL_WasInit(SDL_INIT_VIDEO) != 1)
	{
		if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0)
		{
			VK_ERROR("Failed to initialize SDL video subsystem: %s", SDL_GetError());
			return;
		}
	}

	// Initialize events subsystem
	if (SDL_WasInit(SDL_INIT_EVENTS) != 1)
	{
		if (SDL_InitSubSystem(SDL_INIT_EVENTS) != 0)
		{
			LOG_ERROR("Failed to initialize SDL events subsystem: %s", SDL_GetError());
			return;
		}
	}

	int num_displays = SDL_GetNumVideoDisplays();
	for (int i = 0; i < num_displays; i++)
	{
		SDL_DisplayMode display_mode;
		SDL_GetCurrentDisplayMode(i, &display_mode);

		VideoMode mode;
		mode.width        = display_mode.w;
		mode.height       = display_mode.h;
		mode.refresh_rate = display_mode.refresh_rate;
		mode.red_bits = mode.green_bits = mode.blue_bits = SDL_BITSPERPIXEL(display_mode.format) / 3;
		m_monitors.push_back(Monitor(mode));
	}

	// Create SDL window
	m_window = SDL_CreateWindow(
	    ("IlumEngine v" + std::string(ENGINE_VERSION)).c_str(),
	    m_monitors[0].GetWidth() / 8,
	    m_monitors[0].GetHeight() / 8,
	    m_monitors[0].GetWidth() * 6 / 8,
	    m_monitors[0].GetHeight() * 6 / 8,
	    SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_VULKAN);

	if (!m_window)
	{
		LOG_ERROR("Could not create window: %s", SDL_GetError());
		return;
	}
}

Window::~Window()
{
	// Destroy SDL window
	SDL_DestroyWindow(m_window);

	// Shutdown SDL window
	SDL_Quit();
}

void Window::OnUpdate()
{
	PollEvent();
}

uint32_t Window::GetWidth() const
{
	int w, h;
	SDL_GetWindowSize(m_window, &w, &h);
	return static_cast<uint32_t>(w);
}

uint32_t Window::GetHeight() const
{
	int w, h;
	SDL_GetWindowSize(m_window, &w, &h);
	return static_cast<uint32_t>(h);
}

uint32_t Window::GetPositionX() const
{
	int x, y;
	SDL_GetWindowPosition(m_window, &x, &y);
	return static_cast<uint32_t>(x);
}

uint32_t Window::GetPositionY() const
{
	int x, y;
	SDL_GetWindowPosition(m_window, &x, &y);
	return static_cast<uint32_t>(y);
}

const std::string Window::GetTitle() const
{
	return SDL_GetWindowTitle(m_window);
}

SDL_Window* Window::GetHandle() const
{
	return m_window;
}

void Window::SetSize(uint32_t width, uint32_t height)
{
	SDL_SetWindowSize(m_window, static_cast<int>(width), static_cast<int>(height));
}

void Window::SetPosition(uint32_t x, uint32_t y)
{
	SDL_SetWindowPosition(m_window, static_cast<int>(x), static_cast<int>(y));
}

void Window::SetTitle(const std::string& title)
{
	SDL_SetWindowTitle(m_window, title.c_str());
}

void Window::SetIcon(const std::string& filepath)
{
	auto icon = SDL_LoadBMP(filepath.c_str());
	if (icon)
	{
		SDL_SetColorKey(icon, true, SDL_MapRGB(icon->format, 0, 0, 0));
		SDL_SetWindowIcon(m_window, icon);
		SDL_FreeSurface(icon);
	}
	else
	{
		LOG_ERROR("Failed to load window icon! (%s)\n", SDL_GetError());
	}
}

void Window::SetCursor(bool enable)
{
	SDL_ShowCursor(enable);
}

void Window::Minimize()
{
	SDL_MinimizeWindow(m_window);
}

void Window::Maximize()
{
	SDL_MaximizeWindow(m_window);
}

void Window::Fullscreen()
{
	SDL_SetWindowFullscreen(m_window, SDL_WINDOW_FULLSCREEN);
}

void Window::FullscreenBorderless()
{
	SDL_SetWindowFullscreen(m_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
}

bool Window::ShouldClose()
{
	return m_close;
}

void Window::Show()
{
	SDL_ShowWindow(m_window);
}

void Window::Hide()
{
	SDL_HideWindow(m_window);
}

void Window::Focus()
{
	SDL_RaiseWindow(m_window);
}

bool Window::IsHidden() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_HIDDEN;
}

bool Window::IsFocused() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_INPUT_FOCUS;
}

bool Window::IsShown() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_SHOWN;
}

bool Window::IsMinimized() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_MINIMIZED;
}

bool Window::IsMaximized() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_MAXIMIZED;
}

bool Window::IsFullscreen() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_FULLSCREEN;
}

bool Window::IsFullscreenBorderless() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_FULLSCREEN_DESKTOP;
}

void Window::PollEvent()
{
	SDL_Event sdl_event;

	while (SDL_PollEvent(&sdl_event))
	{
		if (sdl_event.type == SDL_WINDOWEVENT)
		{
			if (sdl_event.window.event == SDL_WINDOWEVENT_CLOSE)
			{
				Event_Close.Invoke();
				m_close = true;
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_RESIZED)
			{
				Event_Resize.Invoke(GetWidth(), GetHeight());
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_MOVED)
			{
				Event_Move.Invoke(GetWidth(), GetHeight());
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_MINIMIZED)
			{
				Event_Minimize.Invoke();
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_MAXIMIZED)
			{
				Event_Maximize.Invoke();
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED)
			{
				Event_GainFocus.Invoke();
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_FOCUS_LOST)
			{
				Event_LostFocus.Invoke();
			}
		}
		Event_SDL.Invoke(sdl_event);
	}
}
}        // namespace Ilum::Graphics