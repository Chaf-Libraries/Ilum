#pragma once

#include "Widget.hpp"

#include <RenderGraph/RenderGraph.hpp>

struct ImNodesEditorContext;

namespace Ilum
{
class RenderGraphEditor : public Widget
{
  public:
	RenderGraphEditor(Editor *editor);

	~RenderGraphEditor();

	virtual void Tick() override;

  private:
	void DrawMenu();

  private:
	enum class ResourcePinType
	{
		None,
		PassOut,
		TextureWrite,
		TextureRead,
		BufferWrite,
		BufferRead
	};

	enum class PassPinType
	{
		None,
		PassIn,
		PassTexture,
		PassBuffer
	};

	int32_t GetPinID(const RGHandle &handle, ResourcePinType pin);

	int32_t GetPinID(RGHandle *handle, PassPinType pin);

	bool ValidLink(ResourcePinType resource, PassPinType pass);

  private:
	ImNodesEditorContext* m_context = nullptr;

	RenderGraphDesc m_desc;

	bool m_need_compile = false;

	size_t m_current_handle = 0;

	std::unordered_map<int32_t, std::pair<RGHandle *, PassPinType>>   m_pass_pin;
	std::unordered_map<int32_t, std::pair<RGHandle, ResourcePinType>> m_resource_pin;
};
}        // namespace Ilum