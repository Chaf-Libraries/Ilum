#include "GLFWWindow.hpp"
#include "../Event/ApplicationEvent.hpp"
#include "../Event/KeyEvent.hpp"
#include "../Event/MouseEvent.hpp"
#include "../Input.hpp"
#include "../Logger.hpp"

#ifdef _WIN32
#	define GLFW_EXPOSE_NATIVE_WIN32
#endif        // _WIN32

#include <GLFW/glfw3native.h>

#include <cassert>

namespace Ilum::Core
{
#ifdef USE_GLFW
std::function<Window *(const WindowDesc &)> Window::CreateFunc = [](const WindowDesc &desc) -> GLFWWindow * { return new GLFWWindow(desc); };
#endif        // USE_GLFW

inline void GLFWErrorCallback(int32_t error, const char *desc)
{
	LOG_ERROR("GLFW Error {}: {}", error, desc);
}

GLFWWindow::GLFWWindow(const WindowDesc &desc)
{
	m_window_data.desc = desc;

	if (!s_initialized)
	{
		assert(glfwInit());

		s_initialized = true;
	}

	m_handle = glfwCreateWindow(static_cast<int>(desc.width), static_cast<int>(desc.height), desc.title.c_str(), nullptr, nullptr);

	glfwSetWindowUserPointer(m_handle, &m_window_data);

	glfwSetWindowSizeCallback(m_handle, [](GLFWwindow *window, int32_t width, int32_t height) {
		auto data         = static_cast<decltype(m_window_data) *>(glfwGetWindowUserPointer(window));
		data->desc.width  = width;
		data->desc.height = height;
		data->callback(WindowResizedEvent(width, height));
	});

	glfwSetWindowCloseCallback(m_handle, [](GLFWwindow *window) {
		auto data = static_cast<decltype(m_window_data) *>(glfwGetWindowUserPointer(window));
		data->callback(WindowClosedEvent());
	});

	glfwSetKeyCallback(m_handle, [](GLFWwindow *window, int key, int scancode, int action, int mods) {
		auto data = static_cast<decltype(m_window_data) *>(glfwGetWindowUserPointer(window));

		switch (action)
		{
			case GLFW_PRESS: {
				data->callback(KeyPressedEvent(key, 0));
				break;
			}
			case GLFW_RELEASE: {
				data->callback(KeyReleasedEvent(key));
				break;
			}
			case GLFW_REPEAT: {
				data->callback(KeyPressedEvent(key, 1));
				break;
			}
			default:
				break;
		}
	});

	glfwSetCharCallback(m_handle, [](GLFWwindow *window, unsigned int keycode) {
		auto data = static_cast<decltype(m_window_data) *>(glfwGetWindowUserPointer(window));
		data->callback(KeyTypedEvent(keycode));
	});

	glfwSetMouseButtonCallback(m_handle, [](GLFWwindow *window, int button, int action, int mods) {
		auto data = static_cast<decltype(m_window_data) *>(glfwGetWindowUserPointer(window));

		switch (action)
		{
			case GLFW_PRESS: {
				data->callback(MouseButtonPressedEvent(button));
				break;
			}
			case GLFW_RELEASE: {
				data->callback(MouseButtonReleasedEvent(button));
				break;
			}
			default:
				break;
		}
	});

	glfwSetScrollCallback(m_handle, [](GLFWwindow *window, double x_offset, double y_offset) {
		auto data = static_cast<decltype(m_window_data) *>(glfwGetWindowUserPointer(window));
		data->callback(MouseScrolledEvent(static_cast<float>(x_offset), static_cast<float>(y_offset)));
	});

	glfwSetCursorPosCallback(m_handle, [](GLFWwindow *window, double x_pos, double y_pos) {
		auto data = static_cast<decltype(m_window_data) *>(glfwGetWindowUserPointer(window));
		data->callback(MouseMovedEvent(static_cast<float>(x_pos), static_cast<float>(y_pos)));
	});
}

GLFWWindow::~GLFWWindow()
{
	glfwDestroyWindow(m_handle);
}

void GLFWWindow::OnUpdate()
{
	glfwPollEvents();

	if (m_window_data.desc.backend == GraphicsBackend::OpenGL)
	{
		glfwSwapBuffers(m_handle);
	}
}

uint32_t GLFWWindow::GetWidth() const
{
	return m_window_data.desc.width;
}

uint32_t GLFWWindow::GetHeight() const
{
	return m_window_data.desc.height;
}

const std::string &GLFWWindow::GetTitle() const
{
	return m_window_data.desc.title;
}

void *GLFWWindow::GetHandle()
{
#ifdef _WIN32
	return glfwGetWin32Window(m_handle);
#else
	return nullptr;
#endif        // _WIN32
}

void GLFWWindow::SetEventCallback(const EventCallbackFunc &callback)
{
	m_window_data.callback = callback;
}

void GLFWWindow::SetVSync(bool enable)
{
	m_window_data.desc.vsync = enable;
	glfwSwapInterval(static_cast<int32_t>(enable));
}

void GLFWWindow::SetIcon(uint8_t *icon, uint32_t width, uint32_t height)
{
	GLFWimage image;
	image.width  = width;
	image.height = height;
	image.pixels = icon;

	glfwSetWindowIcon(m_handle, 1, &image);
}

void GLFWWindow::SetTitle(const std::string &title)
{
	glfwSetWindowTitle(m_handle, title.c_str());
	m_window_data.desc.title = title;
}

void GLFWWindow::SetMousePosition(float x, float y)
{
	glfwSetCursorPos(m_handle, x, y);
}

void GLFWWindow::HideMouse(bool enable)
{
	if (enable)
	{
		glfwSetInputMode(m_handle, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
	}
	else
	{
		glfwSetInputMode(m_handle, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
	}
}
}        // namespace Ilum::Core