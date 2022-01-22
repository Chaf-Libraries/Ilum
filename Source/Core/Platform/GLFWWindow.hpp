#pragma once

#include "../Window.hpp"

#include <GLFW/glfw3.h>

namespace Ilum::Core
{
class GLFWWindow : public Window
{
  public:
	GLFWWindow(const WindowDesc &desc);

	virtual ~GLFWWindow() override;

	virtual void OnUpdate() override;

	virtual uint32_t GetWidth() const override;

	virtual uint32_t GetHeight() const override;

	virtual const std::string &GetTitle() const override;

	virtual void *GetHandle() override;

	virtual void SetEventCallback(const EventCallbackFunc &callback) override;

	virtual void SetVSync(bool enable) override;

	virtual void SetIcon(uint8_t *icon, uint32_t width, uint32_t height) override;

	virtual void SetTitle(const std::string &title) override;

	virtual void SetMousePosition(float x, float y) override;

	virtual void HideMouse(bool enable) override;

  private:
	inline static bool s_initialized = false;

	GLFWwindow *m_handle = nullptr;

	struct
	{
		WindowDesc        desc;
		EventCallbackFunc callback;
	} m_window_data;
};
}        // namespace Ilum::Core