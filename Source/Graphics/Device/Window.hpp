#pragma once

#include "Monitor.hpp"

#include <Core/Event.hpp>

#include <string>
#include <vector>

struct SDL_Window;
union SDL_Event;

namespace Ilum::Graphics
{
class Window
{
  public:
	Window();

	~Window();

	void OnUpdate();

	uint32_t          GetWidth() const;
	uint32_t          GetHeight() const;
	uint32_t          GetPositionX() const;
	uint32_t          GetPositionY() const;
	const std::string GetTitle() const;
	SDL_Window *      GetHandle() const;

	void SetSize(uint32_t width, uint32_t height);
	void SetPosition(uint32_t x, uint32_t y);
	void SetTitle(const std::string &title);
	void SetIcon(const std::string &filepath);
	void SetCursor(bool enable);

	void Minimize();
	void Maximize();

	void Fullscreen();
	void FullscreenBorderless();

	bool ShouldClose();
	void Show();
	void Hide();
	void Focus();

	bool IsHidden() const;
	bool IsFocused() const;
	bool IsShown() const;
	bool IsMinimized() const;
	bool IsMaximized() const;
	bool IsFullscreen() const;
	bool IsFullscreenBorderless() const;

	void PollEvent();

  private:
	SDL_Window *         m_window = nullptr;
	std::vector<Monitor> m_monitors;

	bool m_close = false;

  public:
	inline static Core::Event<const SDL_Event &>  Event_SDL;
	inline static Core::Event<uint32_t, uint32_t> Event_Resize;
	inline static Core::Event<uint32_t, uint32_t> Event_Move;
	inline static Core::Event<>                   Event_Minimize;
	inline static Core::Event<>                   Event_Maximize;
	inline static Core::Event<>                   Event_GainFocus;
	inline static Core::Event<>                   Event_LostFocus;
	inline static Core::Event<>                   Event_Close;
};
}        // namespace Ilum::Graphics