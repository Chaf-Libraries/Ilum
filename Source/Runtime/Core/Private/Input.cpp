//#include "Input.hpp"
//
//#include <GLFW/glfw3.h>
//
//namespace Ilum
//{
//void Input::Bind(Window *window)
//{
//	m_window = window;
//}
//
//bool Input::IsKeyPressed(KeyCode keycode)
//{
//	auto state = glfwGetKey(static_cast<GLFWwindow *>(m_window->GetHandle()), static_cast<int32_t>(keycode));
//	return state == GLFW_PRESS || state == GLFW_REPEAT;
//}
//
//bool Input::IsMouseButtonPressed(MouseCode button)
//{
//	auto state = glfwGetMouseButton(static_cast<GLFWwindow *>(m_window->GetHandle()), static_cast<int32_t>(button));
//	return state == GLFW_PRESS;
//}
//
//glm::vec2 Input::GetMousePosition()
//{
//	double xpos, ypos;
//	glfwGetCursorPos(static_cast<GLFWwindow *>(m_window->GetHandle()), &xpos, &ypos);
//	return glm::vec2{float(xpos), float(ypos)};
//}
//
//void Input::SetCursorPosition(const glm::vec2 &pos)
//{
//	glfwSetCursorPos(static_cast<GLFWwindow *>(m_window->GetHandle()), static_cast<double>(pos.x), static_cast<double>(pos.y));
//}
//}        // namespace Ilum