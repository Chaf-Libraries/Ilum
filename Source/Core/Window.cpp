#include "Window.hpp"

namespace Ilum::Core
{
std::shared_ptr<Window> Window::s_instance = nullptr;

Window ::~Window()
{
}

std::shared_ptr<Window> Window::Create(const WindowDesc &desc)
{
	return CreateFunc(desc);
}

void Window::SetInstance(std::shared_ptr<Window> window)
{
	s_instance = window;
}

std::shared_ptr<Window> Window::GetInstance()
{
	return s_instance;
}
}        // namespace Ilum::Core