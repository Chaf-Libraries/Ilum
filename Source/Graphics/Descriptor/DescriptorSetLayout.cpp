#include "DescriptorSetLayout.hpp"
#include "../Device/Device.hpp"
#include "../Shader/SpirvReflection.hpp"

namespace Ilum::Graphics
{
inline VkDescriptorType GetDescriptorType(ReflectionData::Image::Type type)
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

inline VkDescriptorType GetDescriptorType(ReflectionData::Buffer::Type type)
{
	switch (type)
	{
		case ReflectionData::Buffer::Type::None:
			break;
		case ReflectionData::Buffer::Type::Uniform:
			return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		case ReflectionData::Buffer::Type::Storage:
			return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		default:
			break;
	}

	return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

DescriptorSetLayout::DescriptorSetLayout(const Device &device, const ReflectionData &reflection_data, uint32_t set):
    m_device(device), m_set(set)
{
	assert(reflection_data.sets.find(set) != reflection_data.sets.end());

	auto &buffers           = reflection_data.buffers;
	auto &images            = reflection_data.images;
	auto &input_attachments = reflection_data.input_attachments;
	bool  bindless          = false;

	std::vector<VkDescriptorBindingFlags> descriptor_binding_flags = {};

	// Buffer descriptor
	for (const auto &buffer : buffers)
	{
		if (buffer.set != m_set)
		{
			continue;
		}

		auto type = GetDescriptorType(buffer.type);

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
		if (image.set != m_set)
		{
			continue;
		}

		auto type = GetDescriptorType(image.type);

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
		if (input_attachment.set != m_set)
		{
			continue;
		}

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

	VkDescriptorSetLayoutBindingFlagsCreateInfo descriptor_set_layout_binding_flag_create_info = {};
	descriptor_set_layout_binding_flag_create_info.sType                                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;

	descriptor_set_layout_binding_flag_create_info.bindingCount  = static_cast<uint32_t>(descriptor_binding_flags.size());
	descriptor_set_layout_binding_flag_create_info.pBindingFlags = descriptor_binding_flags.data();
	descriptor_set_layout_create_info.pNext                      = bindless ? &descriptor_set_layout_binding_flag_create_info : nullptr;

	vkCreateDescriptorSetLayout(m_device, &descriptor_set_layout_create_info, nullptr, &m_handle);
}

DescriptorSetLayout::~DescriptorSetLayout()
{
	if (m_handle)
	{
		vkDestroyDescriptorSetLayout(m_device, m_handle, nullptr);
	}
}

DescriptorSetLayout::operator const VkDescriptorSetLayout &() const
{
	return m_handle;
}

const VkDescriptorSetLayout &DescriptorSetLayout::GetHandle() const
{
	return m_handle;
}

uint32_t DescriptorSetLayout::GetSet() const
{
	return m_set;
}

const std::vector<VkDescriptorSetLayoutBinding> &DescriptorSetLayout::GetBinding() const
{
	return m_bindings;
}
}        // namespace Ilum::Graphics