#pragma once

#include "RenderNode.hpp"

#include <Graphics/Resource/Buffer.hpp>
#include <Graphics/Resource/Image.hpp>
#include <Graphics/Resource/Sampler.hpp>

#include <map>
#include <unordered_set>

namespace Ilum::Render
{
class IPassNode;
class RenderGraph;

class IResourceNode : public RenderNode
{
	friend class RenderGraph;

  public:
	IResourceNode(const std::string &name, RenderGraph &render_graph);
	~IResourceNode() = default;

	virtual void OnImGui()  = 0;
	virtual void OnImNode() = 0;
	virtual void OnUpdate() = 0;

	bool ReadBy(IPassNode *pass, int32_t pin)
	{
		if (_ReadBy(pass, pin))
		{
			m_read_passes.emplace(pass, pin);
			return true;
		}
		return false;
	}

	bool WriteBy(IPassNode *pass, int32_t pin)
	{
		if (_WriteBy(pass, pin))
		{
			m_write_passes.emplace(pass, pin);
			return true;
		}
		return false;
	}

  protected:
	virtual bool _ReadBy(IPassNode *pass, int32_t pin)  = 0;
	virtual bool _WriteBy(IPassNode *pass, int32_t pin) = 0;

  protected:
	// Pass - Pin
	std::map<IPassNode *, int32_t> m_read_passes;
	std::map<IPassNode *, int32_t> m_write_passes;

	const int32_t  m_write_id;
	const int32_t m_read_id;

	bool m_dirty = false;
};
}        // namespace Ilum::Render