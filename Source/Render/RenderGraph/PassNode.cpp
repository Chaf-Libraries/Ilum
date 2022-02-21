#include "PassNode.hpp"
#include "ResourceNode.hpp"

namespace Ilum::Render
{
	inline VkDescriptorType BufferUsageToDescriptorType(VkBufferUsageFlagBits usage)
	{
	    switch (usage)
	    {
		    case VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT:
			    return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		    case VK_BUFFER_USAGE_STORAGE_BUFFER_BIT:
			    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		    default:
			    break;
	    }

		return VK_DESCRIPTOR_TYPE_MAX_ENUM;
    }

IPassNode::IPassNode(const std::string &name, PassType pass_type) :
    m_name(name), m_type(pass_type)
{
}

bool IPassNode::Bind(uint32_t set, uint32_t binding, IResourceNode *resource, Graphics::ImageReference image)
{
	return false;
}

bool IPassNode::Bind(uint32_t set, uint32_t binding, IResourceNode *resource, const std::vector<Graphics::ImageReference> &images)
{
	return false;
}

bool IPassNode::Bind(uint32_t set, uint32_t binding, IResourceNode* resource, Graphics::BufferReference buffer)
{
	return false;
}

bool IPassNode::Bind(uint32_t set, uint32_t binding, IResourceNode* resource, const std::vector<Graphics::BufferReference>& buffers)
{
	return false;
}

bool IPassNode::Bind(uint32_t set, uint32_t binding, IResourceNode* resource, Graphics::SamplerReference sampler)
{
	return false;
}

bool IPassNode::Bind(uint32_t set, uint32_t binding, IResourceNode* resource, const std::vector<Graphics::SamplerReference>& samplers)
{
	return false;
}

bool IPassNode::Unbind(IResourceNode* resource)
{
	return false;
}

bool IPassNode::Validate()
{
	// Check binding
	size_t bind_resource_count = 0;
	for (auto& [set, bindings] : m_resource_bind)
	{
		bind_resource_count += bindings.size();
	}

	if (m_buffer_bind_infos.size() + m_image_bind_infos.size() + m_sampler_bind_infos.size() != bind_resource_count)
	{
		return false;
	}

	// Check attachment
	if (m_attachments.size() != m_resource_attachment.size())
	{
		return false;
	}
}
}        // namespace Ilum::Render