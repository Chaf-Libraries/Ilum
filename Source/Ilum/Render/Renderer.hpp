#pragma once

#include "RGBuilder.hpp"
#include "RenderGraph.hpp"

#include <RHI/Device.hpp>
#include <RHI/ImGuiContext.hpp>
#include <RHI/Texture.hpp>

namespace Ilum
{
class Scene;
class AABB;

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

	const VkExtent2D &GetExtent() const;
	const VkExtent2D &GetViewport() const;

	Scene *GetScene();

	void SetScene(Scene *scene);
	void SetPresent(Texture *present);
	void SetDepthStencil(Texture *depth_stencil);

  private:
	void CreateSampler();

  private:
	void KullaContyApprox();
	void GenerateLUT();
	void MousePicking(uint32_t x, uint32_t y);

  private:
	RHIDevice *p_device = nullptr;
	Scene     *p_scene  = nullptr;

	RGBuilder   m_rg_builder;
	RenderGraph m_rg;

	VkExtent2D m_extent          = {1920, 1080};
	VkExtent2D m_viewport        = {1920, 1080};
	bool       m_viewport_update = false;

	Texture *p_present = nullptr;
	Texture *p_depth_stencil = nullptr;

	std::unique_ptr<Buffer> m_picking_buffer = nullptr;

	// LUT
	std::array<std::unique_ptr<Texture>, 2> m_precomputes;

	std::unique_ptr<Texture> m_ggx_lut     = nullptr;
	std::unique_ptr<Texture> m_charlie_lut = nullptr;

	// Sampler
	std::array<std::unique_ptr<Sampler>, 8> m_samplers;

  private:
	void DrawPrimitive();
	void DrawAABB(CommandBuffer &cmd_buffer, const std::vector<std::pair<AABB, glm::vec3>> &aabbs);
	void DrawBVH(CommandBuffer &cmd_buffer);

  private:
	glm::vec3 m_translate_velocity = glm::vec3(0.f);
	glm::vec2 m_last_position      = glm::vec2(0.f);
};
}        // namespace Ilum