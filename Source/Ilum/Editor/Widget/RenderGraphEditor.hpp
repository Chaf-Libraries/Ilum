#pragma once

#include "Widget.hpp"

#include <RenderCore/RenderGraph/RenderGraph.hpp>

namespace Ilum
{
class RenderGraphEditor : public Widget
{
  public:
	RenderGraphEditor(Editor *editor);

	~RenderGraphEditor();

	virtual void Tick() override;

  private:
	enum class PinType
	{
		PassIn,
		PassOut,
		PassTexture,
		PassBuffer,
		TextureWrite,
		TextureRead,
		BufferWrite,
		BufferRead
	};

	int32_t GetPinID(const RGHandle *handle, PinType pin);

	bool ValidLink(PinType src, PinType dst);

  private:
	RenderGraphDesc m_desc;

	bool m_need_compile = false;

	size_t m_current_handle = 0;

	std::unordered_map<int32_t, std::pair<const RGHandle *, PinType>> m_pin_map;
};
}        // namespace Ilum