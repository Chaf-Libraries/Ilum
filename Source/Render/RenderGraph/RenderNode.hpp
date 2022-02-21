#pragma once

#include <cstdint>

namespace Ilum::Render
{
struct PinDesc
{
	virtual size_t Hash() = 0;
	const int32_t  node;
};

class RenderNode
{
  public:
	RenderNode();
	~RenderNode() = default;

	uint64_t GetUUID() const;

	virtual void OnImGui() = 0;
	virtual void OnImNode() = 0;

  protected:
	uint64_t       NewUUID();
	const uint64_t m_uuid;
};
}        // namespace Ilum::Render