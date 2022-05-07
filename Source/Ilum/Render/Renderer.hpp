#pragma once

#include "RGBuilder.hpp"
#include "RenderGraph.hpp"

#include <RHI/Device.hpp>
#include <RHI/ImGuiContext.hpp>
#include <RHI/Texture.hpp>

namespace Ilum
{
class Renderer
{
  public:
	Renderer(RHIDevice *device);
	~Renderer();

	void Tick();

	void OnImGui(ImGuiContext &context);

  private:
	void CreateSampler();

  private:
	void KullaContyApprox();
	void BRDFPreIntegration();

  private:
	RHIDevice *p_device = nullptr;

	RGBuilder m_rg_builder;
	RenderGraph m_rg;

	// LUT
	std::unique_ptr<Texture> m_kulla_conty_EmuLut  = nullptr;
	std::unique_ptr<Texture> m_kulla_conty_EavgLut = nullptr;
	std::unique_ptr<Texture> m_brdf_preintegration = nullptr;

	// Sampler
	std::unique_ptr<Sampler> m_point_clamp_sampler       = nullptr;
	std::unique_ptr<Sampler> m_point_warp_sampler        = nullptr;
	std::unique_ptr<Sampler> m_bilinear_clamp_sampler    = nullptr;
	std::unique_ptr<Sampler> m_bilinear_warp_sampler     = nullptr;
	std::unique_ptr<Sampler> m_trilinear_clamp_sampler   = nullptr;
	std::unique_ptr<Sampler> m_trilinear_warp_sampler    = nullptr;
	std::unique_ptr<Sampler> m_anisptropic_warp_sampler  = nullptr;
	std::unique_ptr<Sampler> m_anisptropic_clamp_sampler = nullptr;
};
}        // namespace Ilum