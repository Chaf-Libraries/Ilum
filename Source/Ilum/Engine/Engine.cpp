#include "Engine.hpp"
#include "Context.hpp"
#include "Device/Input.hpp"
#include "Device/Window.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "ImGui/ImGuiContext.hpp"
#include "Renderer/Renderer.hpp"
#include "Threading/ThreadPool.hpp"
#include "Timing/Timer.hpp"

namespace Ilum
{
Engine *Engine::s_instance = nullptr;

Engine::Engine()
{
	s_instance = this;

	m_context = createScope<Context>();

	m_context->addSubsystem<Timer>();
	m_context->addSubsystem<ThreadPool>();
	m_context->addSubsystem<Window>();
	m_context->addSubsystem<Input>();
	m_context->addSubsystem<GraphicsContext>();
	m_context->addSubsystem<Renderer>();
}

Engine::~Engine()
{
	m_context->onShutdown();
}

Engine *Engine::instance()
{
	return s_instance;
}

void Engine::onInitialize()
{
	m_context->onInitialize();
}

void Engine::onTick()
{
	m_context->onPreTick();

	auto *timer = m_context->getSubsystem<Timer>();

	m_context->onTick(TickType::Smoothed, static_cast<float>(timer->getDeltaTimeSecondSmoothed()));
	m_context->onTick(TickType::Variable, static_cast<float>(timer->getDeltaTimeSecond()));

	m_context->onPostTick();
}

Context &Engine::getContext()
{
	return *m_context;
}
}        // namespace Ilum