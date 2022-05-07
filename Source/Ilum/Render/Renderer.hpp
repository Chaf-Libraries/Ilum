#pragma once

#include "RGBuilder.hpp"
#include "RenderGraph.hpp"

#include <RHI/Device.hpp>
#include <RHI/ImGuiContext.hpp>
#include <RHI/Texture.hpp>

namespace Ilum
{
class Scene;

enum class SamplerType : size_t
{
	PointClamp,
	PointWarp,
	BilinearClamp,
	BilinearWarp,
	TrilinearClamp,
	TrilinearWarp,
	AnisptropicClamp,
	AnisptropicWarp,
	MAX_NUM
};

enum class PrecomputeType
{
	KullaContyEnergy,
	KullaContyAverage,
	BRDFPreIntegration,
	MAX_NUM
};

class Renderer
{
  public:
	Renderer(RHIDevice *device, Scene *p_scene = nullptr);
	~Renderer();

	void Tick();

	void OnImGui(ImGuiContext &context);

	Sampler &GetSampler(SamplerType type);

	Texture &GetPrecompute(PrecomputeType type);

	const VkExtent2D GetExtent() const;

	Scene *GetScene();

	void SetScene(Scene *scene);

	void SetPresent(Texture *present);

  private:
	void CreateSampler();

  private:
	void KullaContyApprox();
	void BRDFPreIntegration();

  private:
	RHIDevice *p_device = nullptr;
	Scene     *p_scene  = nullptr;

	RGBuilder   m_rg_builder;
	RenderGraph m_rg;

	VkExtent2D m_extent   = {1920, 1080};
	VkExtent2D m_viewport = {};
	bool       m_viewport_update = false;

	Texture *p_present = nullptr;

	// LUT
	std::array<std::unique_ptr<Texture>, 3> m_precomputes;

	// Sampler
	std::array<std::unique_ptr<Sampler>, 8> m_samplers;
};
}        // namespace Ilum