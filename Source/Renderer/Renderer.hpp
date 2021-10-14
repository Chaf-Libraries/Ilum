#pragma once

#include "Utils/PCH.hpp"

#include "Engine/Subsystem.hpp"
#include "Engine/Context.hpp"

#include "RenderGraph/RenderGraphBuilder.hpp"

namespace Ilum
{
class Renderer: public TSubsystem<Renderer>
{
  public:
	Renderer(Context *context = nullptr);

	~Renderer();

	virtual bool onInitialize() override;

	virtual void onTick(float) override;

	RenderGraphBuilder &getRenderGraphBuilder();

	const RenderGraph &getRenderGraph() const;

	void rebuild();

  private:
	RenderGraphBuilder m_rg_builder;
	scope<RenderGraph> m_render_graph = nullptr;
};
}        // namespace Ilum