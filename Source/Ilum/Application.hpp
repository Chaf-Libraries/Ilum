#pragma once

#include <Core/Window.hpp>

#include <RHI/Device.hpp>

#include <Render/Renderer.hpp>

#include <Asset/AssetManager.hpp>
#include <Asset/Material.hpp>

#include <Scene/Scene.hpp>

namespace Ilum
{
class Application
{
  public:
	Application();
	~Application() = default;

	void Tick();

  private:
	Window       m_window;
	RHIDevice    m_device;
	ImGuiContext m_imgui_context;

	std::unique_ptr<Renderer>     m_renderer      = nullptr;
	std::unique_ptr<AssetManager> m_asset_manager = nullptr;
	std::unique_ptr<Scene>        m_scene         = nullptr;
};
}        // namespace Ilum