#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Context.hpp"
#include "Engine/Subsystem.hpp"

#include "Graphics/Image/Image.hpp"
#include "Graphics/Image/Sampler.hpp"

#include "RenderGraph/RenderGraphBuilder.hpp"

namespace Ilum
{
class Renderer : public TSubsystem<Renderer>
{
  public:
	static VkExtent2D RenderTargetSize;

	struct ImageData
	{
		ImageReference image;
		uint32_t       texID;
	};

  public:
	Renderer(Context *context = nullptr);

	~Renderer();

	virtual bool onInitialize() override;

	virtual void onPostTick() override;

	const RenderGraph &getRenderGraph() const;

	void resetBuilder();

	void rebuild();

  public:
	std::function<void(RenderGraphBuilder &)> buildRenderGraph = nullptr;

  private:
	std::function<void(RenderGraphBuilder &)> defaultBuilder;

  private:
	RenderGraphBuilder m_rg_builder;
	scope<RenderGraph> m_render_graph = nullptr;

	std::vector<Image> m_external_textures;
};
}        // namespace Ilum