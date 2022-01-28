#pragma once

#include "Event/Event.hpp"

#include <functional>
#include <memory>
#include <string>

namespace Ilum::Core
{
enum class GraphicsBackend
{
	Vulkan,
	OpenGL,
	// DirectX12
};

struct WindowDesc
{
	std::string     title      = "window";
	uint32_t        width      = 256;
	uint32_t        height     = 256;
	bool            fullscreen = false;
	bool            vsync      = false;
	bool            borderless = false;
	bool            console    = true;
	GraphicsBackend backend;
};

class Window
{
  public:
	using EventCallbackFunc = std::function<void(const Event &)>;

	static std::shared_ptr<Window> Create(const WindowDesc &desc);

	static void SetInstance(std::shared_ptr<Window> window);

	static std::shared_ptr<Window> GetInstance();

	virtual ~Window();

	virtual void OnUpdate() = 0;

	virtual uint32_t GetWidth() const = 0;

	virtual uint32_t GetHeight() const = 0;

	virtual const std::string &GetTitle() const = 0;

	// Native window handle
	virtual void *GetHandle() = 0;

	virtual void SetEventCallback(const EventCallbackFunc &callback) = 0;

	virtual void SetVSync(bool enable) = 0;

	virtual void SetIcon(uint8_t *icon, uint32_t width, uint32_t height) = 0;

	virtual void SetTitle(const std::string &title) = 0;

	virtual void SetMousePosition(float x, float y) = 0;

	virtual void HideMouse(bool enable) = 0;

  protected:
	static std::function<std::shared_ptr<Window>(const WindowDesc &)> CreateFunc;

	static std::shared_ptr<Window> s_instance;
};
}        // namespace Ilum::Core