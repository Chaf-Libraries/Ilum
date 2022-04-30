#include "Window.hpp"

namespace Ilum
{
Window::Window(const std::string &title, const std::string &icon, uint32_t width, uint32_t height) :
    m_width(width), m_height(height), m_title(title)
{
	if (!glfwInit())
	{
		return;
	}

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
	m_handle = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
	if (!m_handle)
	{
		glfwTerminate();
		return;
	}

	glfwSetWindowUserPointer(m_handle, this);

	glfwSetKeyCallback(m_handle, [](GLFWwindow *window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
		Window *handle = (Window *) glfwGetWindowUserPointer(window);
		handle->OnKeyFunc.Invoke(key, scancode, action, mods);
	});

	glfwSetCharCallback(m_handle, [](GLFWwindow *window, uint32_t codepoint) {
		Window *handle = (Window *) glfwGetWindowUserPointer(window);
		handle->OnCharFunc.Invoke(codepoint);
	});

	glfwSetCharModsCallback(m_handle, [](GLFWwindow *window, uint32_t codepoint, int32_t mods) {
		Window *handle = (Window *) glfwGetWindowUserPointer(window);
		handle->OnCharModsFunc.Invoke(codepoint, mods);
	});

	glfwSetMouseButtonCallback(m_handle, [](GLFWwindow *window, int32_t button, int32_t action, int32_t mods) {
		Window *handle = (Window *) glfwGetWindowUserPointer(window);
		handle->OnMouseButtonFunc.Invoke(button, action, mods);
	});

	glfwSetCursorPosCallback(m_handle, [](GLFWwindow *window, double xpos, double ypos) {
		Window *handle = (Window *) glfwGetWindowUserPointer(window);
		handle->OnCursorPosFunc.Invoke(xpos, ypos);
	});

	glfwSetCursorEnterCallback(m_handle, [](GLFWwindow *window, int32_t entered) {
		Window *handle = (Window *) glfwGetWindowUserPointer(window);
		handle->OnCursorEnterFunc.Invoke(entered);
	});

	glfwSetScrollCallback(m_handle, [](GLFWwindow *window, double xoffset, double yoffset) {
		Window *handle = (Window *) glfwGetWindowUserPointer(window);
		handle->OnScrollFunc.Invoke(xoffset, yoffset);
	});

	glfwSetDropCallback(m_handle, [](GLFWwindow *window, int32_t count, const char **paths) {
		Window *handle = (Window *) glfwGetWindowUserPointer(window);
		handle->OnDropFunc.Invoke(count, paths);
	});

	glfwSetWindowSizeCallback(m_handle, [](GLFWwindow *window, int32_t width, int32_t height) {
		Window *handle = (Window *) glfwGetWindowUserPointer(window);
		handle->OnWindowSizeFunc.Invoke(width, height);
		handle->m_width = static_cast<uint32_t>(width);
		handle->m_height = static_cast<uint32_t>(height);		
	});

	glfwSetWindowCloseCallback(m_handle, [](GLFWwindow *window) {
		glfwSetWindowShouldClose(window, true);
	});

	glfwSetInputMode(m_handle, GLFW_RAW_MOUSE_MOTION, GLFW_FALSE);
}

Window::~Window()
{
	glfwDestroyWindow(m_handle);
	glfwTerminate();
}

bool Window::Tick()
{
	if (!glfwWindowShouldClose(m_handle))
	{
		glfwPollEvents();
		return true;
	}
	return false;
}

bool Window::IsKeyDown(int32_t key) const
{
	if (key < GLFW_KEY_SPACE || key > GLFW_KEY_LAST)
	{
		return false;
	}
	return glfwGetKey(m_handle, key) == GLFW_PRESS;
}

bool Window::IsMouseButtonDown(int32_t button) const
{
	if (button < GLFW_MOUSE_BUTTON_1 || button > GLFW_MOUSE_BUTTON_LAST)
	{
		return false;
	}

	return glfwGetMouseButton(m_handle, button) == GLFW_PRESS;
}
}        // namespace Ilum