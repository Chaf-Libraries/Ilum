#pragma once

#include "Window.hpp"

namespace Ilum::Core
{
Window *Window::s_instance = nullptr;

Window ::~Window()
{
}

Window *Window::Create(const WindowDesc &desc)
{
	return CreateFunc(desc);
}

void Window::SetInstance(Window* window)
{
	s_instance = window;
}

Window* Window::GetInstance()
{
	return s_instance;
}
}        // namespace Ilum::Core