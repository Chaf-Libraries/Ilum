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

class IResourceNode : public RenderNode
{
  public:
	IResourceNode(const std::string &name);
	~IResourceNode() = default;

	const std::string &GetName() const;

	virtual void OnImGui()  = 0;
	virtual void OnImNode() = 0;
	virtual void OnUpdate() = 0;

	void ReadBy(IPassNode *pass, VkBufferUsageFlagBits usage){};
	void WriteBy(IPassNode *pass, VkBufferUsageFlagBits usage){};
	void ReadBy(IPassNode *pass, VkImageUsageFlagBits usage){};
	void WriteBy(IPassNode *pass, VkImageUsageFlagBits usage){};

  protected:
	std::string m_name;

	std::map<IPassNode *, VkDescriptorType> m_read_passes;
	std::map<IPassNode *, VkDescriptorType> m_write_passes;

	const uint64_t m_node_id;
	const uint64_t m_write_id;
	const uint64_t m_read_id;

	bool m_dirty = false;
};
}        // namespace Ilum::Render