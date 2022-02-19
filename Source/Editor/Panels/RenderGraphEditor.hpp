#pragma once

#include "../Panel.hpp"

#include <RenderGraph/RenderGraph.hpp>

#include <Vulkan/Texture.hpp>

#include <memory>
#include <vector>

namespace Ilum::Editor
{
class RenderGraphEditor : public Panel
{
  public:
	RenderGraphEditor();
	virtual ~RenderGraphEditor() = default;

	virtual void Show() override;
};
}        // namespace Ilum::Editor