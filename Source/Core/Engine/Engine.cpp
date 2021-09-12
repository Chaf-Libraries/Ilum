#include "Engine.hpp"
#include "Context.hpp"

#include "Core/Device/Input.hpp"
#include "Core/Device/Window.hpp"
#include "Core/Engine/Threading/ThreadPool.hpp"
#include "Core/Engine/Timing/Timer.hpp"

namespace Ilum
{
Engine::Engine()
{
	m_context = createScope<Context>();

	m_context->addSubsystem<Timer>();
	m_context->addSubsystem<ThreadPool>();
	m_context->addSubsystem<Window>();
	m_context->addSubsystem<Input>();

	m_context->onInitialize();
}

Engine::~Engine()
{
	m_context->onShutdown();
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