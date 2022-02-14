#include "Engine.hpp"
#include "Context.hpp"
#include "Graphics/GraphicsContext.hpp"
#include "ImGui/ImGuiContext.hpp"
#include "Renderer/Renderer.hpp"
#include "Editor/Editor.hpp"
#include "Scene/Scene.hpp"

#include <Core/JobSystem/JobSystem.hpp>
#include <Core/Timer.hpp>

#include <Graphics/RenderContext.hpp>
#include <Graphics/Device/Window.hpp>
#include <Graphics/Device/Input.hpp>

namespace Ilum
{
Engine *Engine::s_instance = nullptr;

Engine::Engine()
{
	s_instance = this;

	m_context = createScope<Context>();

	m_context->addSubsystem<GraphicsContext>();
	m_context->addSubsystem<Scene>();
	m_context->addSubsystem<Renderer>();
	m_context->addSubsystem<Editor>();

	m_context->onInitialize();

	Core::JobSystem::Initialize();
}

Engine::~Engine()
{
	m_context->onShutdown();
}

Engine *Engine::instance()
{
	return s_instance;
}

void Engine::onTick()
{
	m_context->onPreTick();

	m_timer.OnUpdate();

	Graphics::RenderContext::GetWindow().OnUpdate();
	Graphics::Input::GetInstance().OnUpdate();

	m_context->onTick(TickType::Smoothed, static_cast<float>(m_timer.GetDeltaTime(true)) / 1000.f);
	m_context->onTick(TickType::Variable, static_cast<float>(m_timer.GetDeltaTime()) / 1000.f);

	m_context->onPostTick();
}

Context &Engine::getContext()
{
	return *m_context;
}
}        // namespace Ilum