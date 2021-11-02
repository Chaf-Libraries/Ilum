#include "Window.hpp"

#include <SDL.h>
#include <SDL_syswm.h>

namespace Ilum
{
Window::Window(Context *context):
    TSubsystem<Window>(context)
{
	// Initialize video subsystem
	if (SDL_WasInit(SDL_INIT_VIDEO) != 1)
	{
		if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0)
		{
			LOG_ERROR("Failed to initialize SDL video subsystem: %s", SDL_GetError());
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
		mode.width = display_mode.w;
		mode.height = display_mode.h;
		mode.refresh_rate = display_mode.refresh_rate;
		mode.red_bits = mode.green_bits = mode.blue_bits = SDL_BITSPERPIXEL(display_mode.format) / 3;
		m_monitors.push_back(Monitor(mode));
	}

	// Create SDL window
	m_window = SDL_CreateWindow(
	    ("IlumEngine v" + std::string(ENGINE_VERSION)).c_str(),
	    m_monitors[0].getWidth() / 4,
	    m_monitors[0].getHeight() / 4,
	    m_monitors[0].getWidth() / 2,
	    m_monitors[0].getHeight() / 2,
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

void Window::onPreTick()
{
	pollEvent();
}

uint32_t Window::getWidth() const
{
	int w, h;
	SDL_GetWindowSize(m_window, &w, &h);
	return static_cast<uint32_t>(w);
}

uint32_t Window::getHeight() const
{
	int w, h;
	SDL_GetWindowSize(m_window, &w, &h);
	return static_cast<uint32_t>(h);
}

uint32_t Window::getPositionX() const
{
	int x, y;
	SDL_GetWindowPosition(m_window, &x, &y);
	return static_cast<uint32_t>(x);
}

uint32_t Window::getPositionY() const
{
	int x, y;
	SDL_GetWindowPosition(m_window, &x, &y);
	return static_cast<uint32_t>(y);
}

const std::string Window::getTitle() const
{
	return SDL_GetWindowTitle(m_window);
}

void *Window::getWindowHandle() const
{
	ASSERT(m_window != nullptr);

	SDL_SysWMinfo sys_info;
	SDL_VERSION(&sys_info.version);
	SDL_GetWindowWMInfo(m_window, &sys_info);
#ifdef WIN32
	return static_cast<void *>(sys_info.info.win.window);
#else
	// TODO: Linux support
	return nullptr;
#endif        // WIN32
}

SDL_Window *Window::getSDLHandle() const
{
	return m_window;
}

void Window::setSize(uint32_t width, uint32_t height)
{
	SDL_SetWindowSize(m_window, static_cast<int>(width), static_cast<int>(height));
}

void Window::setPosition(uint32_t x, uint32_t y)
{
	SDL_SetWindowPosition(m_window, static_cast<int>(x), static_cast<int>(y));
}

void Window::setTitle(const std::string &title)
{
	SDL_SetWindowTitle(m_window, title.c_str());
}

void Window::setIcon(const std::string &filepath)
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

void Window::setCursor(bool enable)
{
	SDL_ShowCursor(enable);
}

void Window::minimize()
{
	SDL_MinimizeWindow(m_window);
}

void Window::maximize()
{
	SDL_MaximizeWindow(m_window);
}

void Window::fullscreen()
{
	SDL_SetWindowFullscreen(m_window, SDL_WINDOW_FULLSCREEN);
}

void Window::fullscreenBorderless()
{
	SDL_SetWindowFullscreen(m_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
}

bool Window::shouldClose()
{
	return m_close;
}

void Window::show()
{
	SDL_ShowWindow(m_window);
}

void Window::hide()
{
	SDL_HideWindow(m_window);
}

void Window::focus()
{
	SDL_RaiseWindow(m_window);
}

bool Window::isHidden() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_HIDDEN;
}

bool Window::isFocused() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_INPUT_FOCUS;
}

bool Window::isShown() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_SHOWN;
}

bool Window::isMinimized() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_MINIMIZED;
}

bool Window::isMaximized() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_MAXIMIZED;
}

bool Window::isFullscreen() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_FULLSCREEN;
}

bool Window::isFullscreenBorderless() const
{
	return SDL_GetWindowFlags(m_window) & SDL_WINDOW_FULLSCREEN_DESKTOP;
}

void Window::pollEvent()
{
	SDL_Event sdl_event;

	while (SDL_PollEvent(&sdl_event))
	{
		if (sdl_event.type == SDL_WINDOWEVENT)
		{
			if (sdl_event.window.event == SDL_WINDOWEVENT_CLOSE)
			{
				Event_Close.invoke();
				m_close = true;
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_RESIZED)
			{
				Event_Resize.invoke(getWidth(), getHeight());
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_MOVED)
			{
				Event_Move.invoke(getWidth(), getHeight());
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_MINIMIZED)
			{
				Event_Minimize.invoke();
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_MAXIMIZED)
			{
				Event_Maximize.invoke();
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_FOCUS_GAINED)
			{
				Event_GainFocus.invoke();
			}
			if (sdl_event.window.event == SDL_WINDOWEVENT_FOCUS_LOST)
			{
				Event_LostFocus.invoke();
			}
		}
		Event_SDL.invoke(sdl_event);
	}
}
}        // namespace Ilum