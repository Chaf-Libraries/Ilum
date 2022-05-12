#include "DescriptorState.hpp"
#include "Device.hpp"
#include "PipelineState.hpp"

#include <Core/Macro.hpp>

namespace Ilum
{
VkDescriptorType GetDescriptorType(ShaderReflectionData::Image::Type type)
{
	switch (type)
	{
		case ShaderReflectionData::Image::Type::ImageSampler:
			return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		case ShaderReflectionData::Image::Type::Image:
			return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		case ShaderReflectionData::Image::Type::ImageStorage:
			return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		case ShaderReflectionData::Image::Type::Sampler:
			return VK_DESCRIPTOR_TYPE_SAMPLER;
		default:
			return VK_DESCRIPTOR_TYPE_MAX_ENUM;
	}
}

VkDescriptorType GetDescriptorType(ShaderReflectionData::Buffer::Type type)
{
	switch (type)
	{
		case ShaderReflectionData::Buffer::Type::Uniform:
			return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		case ShaderReflectionData::Buffer::Type::Storage:
			return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		default:
			return VK_DESCRIPTOR_TYPE_MAX_ENUM;
	}
}

VkImageUsageFlagBits GetImageUsage(ShaderReflectionData::Image::Type type)
{
	switch (type)
	{
		case ShaderReflectionData::Image::Type::ImageSampler:
		case ShaderReflectionData::Image::Type::Image:
		case ShaderReflectionData::Image::Type::Sampler:
			return VK_IMAGE_USAGE_SAMPLED_BIT;
		case ShaderReflectionData::Image::Type::ImageStorage:
			return VK_IMAGE_USAGE_STORAGE_BIT;
		default:
			return VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM;
	}
}

DescriptorState::DescriptorState(RHIDevice *device, const PipelineState *pso) :
    p_device(device), p_pso(pso)
{
	m_bind_point = pso->GetBindPoint();

	ShaderReflectionData meta = {};
	for (auto &shader : pso->GetShaders())
	{
		meta += device->ReflectShader(device->LoadShader(shader));
	}

	m_descriptor_sets = p_device->AllocateDescriptorSet(*pso);

	for (auto &set : meta.sets)
	{
		for (auto &image : meta.images)
		{
			if (image.set == set)
			{
				m_image_resolves[set][image.binding].push_back(VkDescriptorImageInfo{VK_NULL_HANDLE, VK_NULL_HANDLE, TextureState(GetImageUsage(image.type)).layout});
				m_resolves[set][image.binding] = VkWriteDescriptorSet{
				    /*sType*/ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				    /*pNext*/ nullptr,
				    /*dstSet*/ m_descriptor_sets[set],
				    /*dstBinding*/ image.binding,
				    /*dstArrayElement*/ 0,
				    /*descriptorCount*/ 0,
				    /*descriptorType*/ GetDescriptorType(image.type),
				    /*pImageInfo*/ nullptr,
				    /*pBufferInfo*/ nullptr,
				    /*pTexelBufferView*/ nullptr};
			}
		}

		for (auto &buffer : meta.buffers)
		{
			if (buffer.set == set)
			{
				m_buffer_resolves[set][buffer.binding].push_back(VkDescriptorBufferInfo{VK_NULL_HANDLE, 0, 0});
				m_resolves[set][buffer.binding] = VkWriteDescriptorSet{
				    /*sType*/ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				    /*pNext*/ nullptr,
				    /*dstSet*/ m_descriptor_sets[set],
				    /*dstBinding*/ buffer.binding,
				    /*dstArrayElement*/ 0,
				    /*descriptorCount*/ 0,
				    /*descriptorType*/ GetDescriptorType(buffer.type),
				    /*pImageInfo*/ nullptr,
				    /*pBufferInfo*/ nullptr,
				    /*pTexelBufferView*/ nullptr};
			}
		}

		for (auto &acceleration_structure : meta.acceleration_structures)
		{
			if (acceleration_structure.set == set)
			{
				m_acceleration_structure_resolves[set][acceleration_structure.binding].push_back(nullptr);
				m_resolves[set][acceleration_structure.binding] = VkWriteDescriptorSet{
				    /*sType*/ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				    /*pNext*/ nullptr,
				    /*dstSet*/ m_descriptor_sets[set],
				    /*dstBinding*/ acceleration_structure.binding,
				    /*dstArrayElement*/ 0,
				    /*descriptorCount*/ 0,
				    /*descriptorType*/ VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
				    /*pImageInfo*/ nullptr,
				    /*pBufferInfo*/ nullptr,
				    /*pTexelBufferView*/ nullptr};
			}
		}
	}
}

DescriptorState &DescriptorState::Bind(uint32_t set, uint32_t binding, Buffer *buffer)
{
	ASSERT(m_buffer_resolves[set][binding].size() == 1);
	if (m_buffer_resolves[set][binding][0].buffer != *buffer)
	{
		m_buffer_resolves[set][binding][0].buffer = *buffer;
		m_buffer_resolves[set][binding][0].range  = buffer->GetSize();
		m_buffer_resolves[set][binding][0].offset = 0;
		m_dirty                                   = true;
	}
	return *this;
}

DescriptorState &DescriptorState::Bind(uint32_t set, uint32_t binding, VkImageView view, VkSampler sampler)
{
	ASSERT(m_image_resolves[set][binding].size() == 1);
	if (m_image_resolves[set][binding][0].imageView != view)
	{
		m_image_resolves[set][binding][0].imageView = view;
		m_image_resolves[set][binding][0].sampler   = sampler;
		m_dirty                                     = true;
	}
	return *this;
}

DescriptorState &DescriptorState::Bind(uint32_t set, uint32_t binding, VkSampler sampler)
{
	ASSERT(m_image_resolves[set][binding].size() == 1);
	ASSERT(m_image_resolves[set][binding][0].imageView == VK_NULL_HANDLE);
	if (m_image_resolves[set][binding][0].sampler != sampler)
	{
		m_image_resolves[set][binding][0].imageView = VK_NULL_HANDLE;
		m_image_resolves[set][binding][0].sampler   = sampler;
		m_dirty                                     = true;
	}
	return *this;
}

DescriptorState &DescriptorState::Bind(uint32_t set, uint32_t binding, AccelerationStructure *acceleration_structure)
{
	ASSERT(m_acceleration_structure_resolves[set][binding].size() == 1);
	if (m_acceleration_structure_resolves[set][binding][0] != acceleration_structure)
	{
		m_acceleration_structure_resolves[set][binding][0] = acceleration_structure;
		m_dirty                                            = true;
	}
	return *this;
}

DescriptorState &DescriptorState::Bind(uint32_t set, uint32_t binding, const std::vector<Buffer *> &buffers)
{
	ASSERT(m_buffer_resolves[set][binding].size() == buffers.size() || m_buffer_resolves[binding].size() == 0);
	if (m_buffer_resolves[set][binding].size() < buffers.size())
	{
		m_buffer_resolves[set][binding].resize(buffers.size());
	}
	for (size_t i = 0; i < buffers.size(); i++)
	{
		if (m_buffer_resolves[set][binding][i].buffer != *buffers[i])
		{
			m_buffer_resolves[set][binding][i].buffer = *buffers[i];
			m_buffer_resolves[set][binding][i].range  = buffers[i]->GetSize();
			m_buffer_resolves[set][binding][i].offset = 0;
			m_dirty                                   = true;
		}
	}
	return *this;
}

DescriptorState &DescriptorState::Bind(uint32_t set, uint32_t binding, const std::vector<VkImageView> &views, VkSampler sampler)
{
	ASSERT(m_image_resolves[set][binding].size() == views.size() || m_image_resolves[binding].size() == 0);
	if (m_image_resolves[set][binding].size() < views.size())
	{
		m_image_resolves[set][binding].resize(views.size());
	}
	for (size_t i = 0; i < views.size(); i++)
	{
		if (m_image_resolves[set][binding][i].imageView != views[i] || m_image_resolves[set][binding][i].sampler != sampler)
		{
			m_image_resolves[set][binding][i].imageView = views[i];
			m_image_resolves[set][binding][i].sampler   = sampler;
			m_dirty                                     = true;
		}
	}
	return *this;
}

DescriptorState &DescriptorState::Bind(uint32_t set, uint32_t binding, const std::vector<VkSampler> &samplers)
{
	ASSERT(m_image_resolves[set][binding].size() == samplers.size() || m_image_resolves[binding].size() == 0);
	if (m_image_resolves[set][binding].size() < samplers.size())
	{
		m_image_resolves[set][binding].resize(samplers.size());
	}
	for (size_t i = 0; i < samplers.size(); i++)
	{
		if (m_image_resolves[set][binding][i].sampler != samplers[i])
		{
			m_image_resolves[set][binding][i].imageView = VK_NULL_HANDLE;
			m_image_resolves[set][binding][i].sampler   = samplers[i];
			m_dirty                                     = true;
		}
	}
	return *this;
}

DescriptorState &DescriptorState::Bind(uint32_t set, uint32_t binding, const std::vector<AccelerationStructure *> &acceleration_structures)
{
	ASSERT(m_acceleration_structure_resolves[set][binding].size() == acceleration_structures.size() || m_acceleration_structure_resolves[binding].size() == 0);
	if (m_acceleration_structure_resolves[set][binding].size() < acceleration_structures.size())
	{
		m_acceleration_structure_resolves[set][binding].resize(acceleration_structures.size());
	}
	for (size_t i = 0; i < acceleration_structures.size(); i++)
	{
		if (m_acceleration_structure_resolves[set][binding][i] != acceleration_structures[i])
		{
			m_acceleration_structure_resolves[set][binding][i] = acceleration_structures[i];
			m_dirty                                            = true;
		}
	}
	return *this;
}

void DescriptorState::Write()
{
	if (!m_dirty)
	{
		return;
	}

	for (auto &[set, descriptor_set] : m_descriptor_sets)
	{
		// Write image
		for (auto &[binding, image] : m_image_resolves[set])
		{
			auto &resolve = m_resolves[set][binding];

			resolve.descriptorCount = static_cast<uint32_t>(image.size());
			resolve.pImageInfo      = image.data();
		}

		// Write buffer
		for (auto &[binding, buffer] : m_buffer_resolves[set])
		{
			auto &resolve = m_resolves[set][binding];

			resolve.descriptorCount = static_cast<uint32_t>(buffer.size());
			resolve.pBufferInfo     = buffer.data();
		}

		// Write acceleration structure
		for (auto &[binding, acceleration_structure] : m_acceleration_structure_resolves[set])
		{
			auto &resolve = m_resolves[set][binding];

			std::vector<VkAccelerationStructureKHR> acceleration_structure_handles(acceleration_structure.size());
			for (size_t i = 0; i < acceleration_structure_handles.size(); i++)
			{
				acceleration_structure_handles[i] = *acceleration_structure[i];
			}

			VkWriteDescriptorSetAccelerationStructureKHR write_descriptor_set_as = {};
			write_descriptor_set_as.sType                                        = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
			write_descriptor_set_as.accelerationStructureCount                   = static_cast<uint32_t>(acceleration_structure_handles.size());
			write_descriptor_set_as.pAccelerationStructures                      = acceleration_structure_handles.data();

			resolve.pNext           = &write_descriptor_set_as;
			resolve.descriptorCount = static_cast<uint32_t>(acceleration_structure_handles.size());
		}

		std::vector<VkWriteDescriptorSet> write_info;
		write_info.reserve(m_resolves[set].size());
		for (auto &write : m_resolves[set])
		{
			write_info.push_back(write.second);
		}
		vkUpdateDescriptorSets(p_device->GetDevice(), static_cast<uint32_t>(write_info.size()), write_info.data(), 0, nullptr);
	}

	m_dirty = false;
}

}        // namespace Ilum