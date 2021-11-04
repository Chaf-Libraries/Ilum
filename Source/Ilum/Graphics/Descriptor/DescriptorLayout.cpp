#include "DescriptorLayout.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Pipeline/Shader.hpp"

#include "Graphics/Vulkan/VK_Debugger.h"

namespace Ilum
{
// TODO: Dynamic buffer?
inline VkDescriptorType find_descriptor_type(ReflectionData::Image::Type type)
{
	switch (type)
	{
		case ReflectionData::Image::Type::None:
			break;
		case ReflectionData::Image::Type::ImageSampler:
			return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		case ReflectionData::Image::Type::Image:
			return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		case ReflectionData::Image::Type::ImageStorage:
			return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		case ReflectionData::Image::Type::Sampler:
			return VK_DESCRIPTOR_TYPE_SAMPLER;
		default:
			break;
	}

	return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

inline VkDescriptorType find_descriptor_type(ReflectionData::Buffer::Type type, bool dynamic)
{
	switch (type)
	{
		case ReflectionData::Buffer::Type::None:
			break;
		case ReflectionData::Buffer::Type::Uniform:
			return dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		case ReflectionData::Buffer::Type::Storage:
			return dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		default:
			break;
	}

	return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

DescriptorLayout::DescriptorLayout(const Shader &shader, const uint32_t set_index) :
    m_set_index(set_index)
{
	auto &buffers           = shader.getReflectionData().buffers;
	auto &images            = shader.getReflectionData().images;
	auto &input_attachments = shader.getReflectionData().input_attachments;
	bool  bindless          = false;

	std::vector<VkDescriptorBindingFlags> descriptor_binding_flags = {};

	// Buffer descriptor
	for (const auto &buffer : buffers)
	{
		if (buffer.set != set_index)
		{
			continue;
		}

		auto type = find_descriptor_type(buffer.type, false);

		VkDescriptorSetLayoutBinding layout_binding = {};
		layout_binding.binding                      = buffer.binding;
		layout_binding.descriptorType               = type;
		layout_binding.stageFlags                   = buffer.stage;
		layout_binding.descriptorCount              = buffer.bindless ? 1024 : buffer.array_size;
		m_bindings.push_back(layout_binding);

		bindless |= buffer.bindless;
		descriptor_binding_flags.push_back(buffer.bindless ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0);
	}

	// Image descriptor
	for (const auto &image : images)
	{
		if (image.set != set_index)
		{
			continue;
		}

		auto type = find_descriptor_type(image.type);
		m_binding_flags.push_back(0);

		VkDescriptorSetLayoutBinding layout_binding = {};
		layout_binding.binding                      = image.binding;
		layout_binding.descriptorType               = type;
		layout_binding.stageFlags                   = image.stage;
		layout_binding.descriptorCount              = image.bindless ? 1024 : image.array_size;
		m_bindings.push_back(layout_binding);

		bindless |= image.bindless;
		descriptor_binding_flags.push_back(image.bindless ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0);
	}

	// Input attachment descriptor
	for (const auto &input_attachment : input_attachments)
	{
		if (input_attachment.set != set_index)
		{
			continue;
		}

		m_binding_flags.push_back(0);

		VkDescriptorSetLayoutBinding layout_binding = {};
		layout_binding.binding                      = input_attachment.binding;
		layout_binding.descriptorType               = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
		layout_binding.stageFlags                   = input_attachment.stage;
		layout_binding.descriptorCount              = input_attachment.bindless ? 1024 : input_attachment.array_size;
		m_bindings.push_back(layout_binding);

		bindless |= input_attachment.bindless;
		descriptor_binding_flags.push_back(input_attachment.bindless ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0);
	}

	// Create descriptor set layout
	VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};

	descriptor_set_layout_create_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptor_set_layout_create_info.bindingCount = static_cast<uint32_t>(m_bindings.size());
	descriptor_set_layout_create_info.pBindings    = m_bindings.data();
	descriptor_set_layout_create_info.flags        = std::find(m_binding_flags.begin(), m_binding_flags.end(), VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT) != m_binding_flags.end() ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0;

	VkDescriptorSetLayoutBindingFlagsCreateInfo descriptor_set_layout_binding_flag_create_info = {};
	descriptor_set_layout_binding_flag_create_info.sType                                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;

	descriptor_set_layout_binding_flag_create_info.bindingCount  = static_cast<uint32_t>(descriptor_binding_flags.size());
	descriptor_set_layout_binding_flag_create_info.pBindingFlags = descriptor_binding_flags.data();
	descriptor_set_layout_create_info.pNext                      = bindless ? &descriptor_set_layout_binding_flag_create_info : nullptr;

	vkCreateDescriptorSetLayout(GraphicsContext::instance()->getLogicalDevice(), &descriptor_set_layout_create_info, nullptr, &m_handle);
}

DescriptorLayout::~DescriptorLayout()
{
	if (m_handle)
	{
		vkDestroyDescriptorSetLayout(GraphicsContext::instance()->getLogicalDevice(), m_handle, nullptr);
		m_handle = VK_NULL_HANDLE;
	}
}

DescriptorLayout::DescriptorLayout(DescriptorLayout &&other) :
    m_handle(other.m_handle),
    m_set_index(other.m_set_index),
    m_bindings(std::move(other.m_bindings)),
    m_binding_flags(std::move(other.m_binding_flags))
{
	other.m_handle = VK_NULL_HANDLE;
}

DescriptorLayout &DescriptorLayout::operator=(DescriptorLayout &&other)
{
	m_handle        = other.m_handle;
	m_set_index     = other.m_set_index;
	m_bindings      = std::move(other.m_bindings);
	m_binding_flags = std::move(other.m_binding_flags);
	other.m_handle  = VK_NULL_HANDLE;
	return *this;
}

const VkDescriptorSetLayout &DescriptorLayout::getDescriptorSetLayout() const
{
	return m_handle;
}

DescriptorLayout::operator const VkDescriptorSetLayout &() const
{
	return m_handle;
}

const std::vector<VkDescriptorSetLayoutBinding> &DescriptorLayout::getBindings() const
{
	return m_bindings;
}

const std::vector<VkDescriptorBindingFlags> &DescriptorLayout::getBindingFlags() const
{
	return m_binding_flags;
}
}        // namespace Ilum