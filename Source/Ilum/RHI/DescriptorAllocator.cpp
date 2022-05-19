#include "DescriptorAllocator.hpp"
#include "Device.hpp"

#include <Core/Macro.hpp>

#include <map>

namespace Ilum
{
inline VkDescriptorType FindDescriptorType(ShaderReflectionData::Image::Type type)
{
	switch (type)
	{
		case ShaderReflectionData::Image::Type::None:
			break;
		case ShaderReflectionData::Image::Type::ImageSampler:
			return VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		case ShaderReflectionData::Image::Type::Image:
			return VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
		case ShaderReflectionData::Image::Type::ImageStorage:
			return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		case ShaderReflectionData::Image::Type::Sampler:
			return VK_DESCRIPTOR_TYPE_SAMPLER;
		default:
			break;
	}

	return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

inline VkDescriptorType FindDescriptorType(ShaderReflectionData::Buffer::Type type, bool dynamic)
{
	switch (type)
	{
		case ShaderReflectionData::Buffer::Type::None:
			break;
		case ShaderReflectionData::Buffer::Type::Uniform:
			return dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		case ShaderReflectionData::Buffer::Type::Storage:
			return dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		default:
			break;
	}

	return VK_DESCRIPTOR_TYPE_MAX_ENUM;
}

DescriptorLayout::DescriptorLayout(RHIDevice *device, const ShaderReflectionData &meta, const uint32_t set_index) :
    p_device(device), m_set_index(set_index)
{
	auto &buffers                 = meta.buffers;
	auto &images                  = meta.images;
	auto &acceleration_structures = meta.acceleration_structures;
	auto &input_attachments       = meta.input_attachments;
	bool  bindless                = false;

	std::vector<VkDescriptorBindingFlags> descriptor_binding_flags = {};

	// Buffer descriptor
	for (const auto &buffer : buffers)
	{
		if (buffer.set != set_index)
		{
			continue;
		}

		auto type = FindDescriptorType(buffer.type, false);

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

		auto type = FindDescriptorType(image.type);
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

	// Acceleration Structure descriptor
	for (const auto &acceleration_structure : acceleration_structures)
	{
		if (acceleration_structure.set != set_index)
		{
			continue;
		}

		m_binding_flags.push_back(0);

		VkDescriptorSetLayoutBinding layout_binding = {};
		layout_binding.binding                      = acceleration_structure.binding;
		layout_binding.descriptorType               = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		layout_binding.stageFlags                   = acceleration_structure.stage;
		layout_binding.descriptorCount              = acceleration_structure.array_size;
		m_bindings.push_back(layout_binding);

		descriptor_binding_flags.push_back(0);
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

	vkCreateDescriptorSetLayout(p_device->GetDevice(), &descriptor_set_layout_create_info, nullptr, &m_handle);
}

DescriptorLayout::~DescriptorLayout()
{
	if (m_handle)
	{
		vkDeviceWaitIdle(p_device->GetDevice());
		vkDestroyDescriptorSetLayout(p_device->GetDevice(), m_handle, nullptr);
	}
}

DescriptorLayout::DescriptorLayout(DescriptorLayout &&other) noexcept :
    p_device(other.p_device),
    m_handle(other.m_handle),
    m_set_index(other.m_set_index),
    m_bindings(std::move(other.m_bindings)),
    m_binding_flags(std::move(other.m_binding_flags))
{
	other.p_device = nullptr;
	other.m_handle = VK_NULL_HANDLE;
}

DescriptorLayout &DescriptorLayout::operator=(DescriptorLayout &&other) noexcept
{
	p_device        = other.p_device;
	m_handle        = other.m_handle;
	m_set_index     = other.m_set_index;
	m_bindings      = std::move(other.m_bindings);
	m_binding_flags = std::move(other.m_binding_flags);
	other.m_handle  = VK_NULL_HANDLE;
	other.p_device  = nullptr;
	return *this;
}

DescriptorLayout::operator VkDescriptorSetLayout() const
{
	return m_handle;
}

const VkDescriptorSetLayout &DescriptorLayout::GetHandle() const
{
	return m_handle;
}

const std::vector<VkDescriptorSetLayoutBinding> &DescriptorLayout::GetBindings() const
{
	return m_bindings;
}

const std::vector<VkDescriptorBindingFlags> &DescriptorLayout::GetBindingFlags() const
{
	return m_binding_flags;
}

DescriptorPool::DescriptorPool(RHIDevice *device, const DescriptorLayout &descriptor_layout, uint32_t pool_size) :
    p_device(device)
{
	const auto &bindings = descriptor_layout.GetBindings();

	std::map<VkDescriptorType, uint32_t> descriptor_type_counts;

	for (auto &binding : bindings)
	{
		descriptor_type_counts[binding.descriptorType] += binding.descriptorCount;
	}

	m_pool_sizes.resize(descriptor_type_counts.size());

	auto pool_size_it = m_pool_sizes.begin();

	for (auto &it : descriptor_type_counts)
	{
		pool_size_it->type            = it.first;
		pool_size_it->descriptorCount = it.second * pool_size;
		++pool_size_it;
	}

	m_pool_max_sets = pool_size;
}

DescriptorPool::~DescriptorPool()
{
	for (auto &pool : m_descriptor_pools)
	{
		vkDestroyDescriptorPool(p_device->GetDevice(), pool, nullptr);
	}
}

DescriptorPool::DescriptorPool(DescriptorPool &&other) noexcept :
    p_device(other.p_device),
    m_pool_index(other.m_pool_index),
    m_pool_max_sets(other.m_pool_max_sets),
    m_pool_sizes(std::move(other.m_pool_sizes)),
    m_descriptor_pools(std::move(other.m_descriptor_pools)),
    m_pool_sets_count(std::move(other.m_pool_sets_count)),
    m_set_pool_mapping(std::move(other.m_set_pool_mapping))
{
	other.p_device = nullptr;
}

DescriptorPool &DescriptorPool::operator=(DescriptorPool &&other) noexcept
{
	p_device           = other.p_device;
	m_pool_index       = other.m_pool_index;
	m_pool_max_sets    = other.m_pool_max_sets;
	m_pool_sizes       = std::move(other.m_pool_sizes),
	m_descriptor_pools = std::move(other.m_descriptor_pools);
	m_pool_sets_count  = std::move(other.m_pool_sets_count);
	m_set_pool_mapping = std::move(other.m_set_pool_mapping);

	other.p_device = nullptr;

	return *this;
}

void DescriptorPool::Reset()
{
	for (auto &pool : m_descriptor_pools)
	{
		vkResetDescriptorPool(p_device->GetDevice(), pool, 0);
	}

	std::fill(m_pool_sets_count.begin(), m_pool_sets_count.end(), 0);
	m_set_pool_mapping.clear();

	m_pool_index = 0;
}

bool DescriptorPool::Has(VkDescriptorSet descriptor_set)
{
	return m_set_pool_mapping.find(descriptor_set) != m_set_pool_mapping.end();
}

VkDescriptorSet DescriptorPool::Allocate(const DescriptorLayout &descriptor_layout)
{
	m_pool_index = FindAvaliablePool(descriptor_layout, m_pool_index);

	++m_pool_sets_count[m_pool_index];

	VkDescriptorSetAllocateInfo allocate_info = {};
	allocate_info.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocate_info.descriptorPool              = m_descriptor_pools[m_pool_index];
	allocate_info.descriptorSetCount          = 1;
	allocate_info.pSetLayouts                 = &descriptor_layout.GetHandle();

	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

	if (vkAllocateDescriptorSets(p_device->GetDevice(), &allocate_info, &descriptor_set) != VK_SUCCESS)
	{
		LOG_FATAL("Failed to allocate descriptor set!");

		--m_pool_sets_count[m_pool_index];
		return VK_NULL_HANDLE;
	}

	m_set_pool_mapping.emplace(descriptor_set, m_pool_index);

	return descriptor_set;
}

void DescriptorPool::Free(VkDescriptorSet descriptor_set)
{
	auto it = m_set_pool_mapping.find(descriptor_set);

	if (it == m_set_pool_mapping.end())
	{
		return;
	}

	auto desc_pool_index = it->second;

	vkFreeDescriptorSets(p_device->GetDevice(), m_descriptor_pools[desc_pool_index], 1, &descriptor_set);

	m_set_pool_mapping.erase(it);

	--m_pool_sets_count[desc_pool_index];

	m_pool_index = desc_pool_index;
}

uint32_t DescriptorPool::FindAvaliablePool(const DescriptorLayout &descriptor_layout, uint32_t pool_index)
{
	// Create a new pool
	if (m_descriptor_pools.size() <= pool_index)
	{
		VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
		descriptor_pool_create_info.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptor_pool_create_info.pPoolSizes                 = m_pool_sizes.data();
		descriptor_pool_create_info.poolSizeCount              = static_cast<uint32_t>(m_pool_sizes.size());
		descriptor_pool_create_info.maxSets                    = m_pool_max_sets;
		descriptor_pool_create_info.flags                      = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

		auto &binding_flags = descriptor_layout.GetBindingFlags();
		for (auto &binding_flag : binding_flags)
		{
			if (binding_flag & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT)
			{
				descriptor_pool_create_info.flags |= VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
			}
		}

		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

		if (vkCreateDescriptorPool(p_device->GetDevice(), &descriptor_pool_create_info, nullptr, &descriptor_pool) != VK_SUCCESS)
		{
			LOG_FATAL("Failed to create descriptor pool!");
			return 0;
		}

		m_descriptor_pools.push_back(descriptor_pool);
		m_pool_sets_count.push_back(0);
		return pool_index;
	}
	else if (m_pool_sets_count[pool_index] < m_pool_max_sets)
	{
		return pool_index;
	}

	return FindAvaliablePool(descriptor_layout, ++pool_index);
}

DescriptorAllocator::DescriptorAllocator(RHIDevice *device) :
    p_device(device)
{
}

VkDescriptorSetLayout DescriptorAllocator::GetDescriptorLayout(const ShaderReflectionData &meta, uint32_t set_index)
{
	size_t hash = meta.Hash();
	HashCombine(hash, set_index);

	if (m_hash_layout_mapping.find(hash) != m_hash_layout_mapping.end())
	{
		return m_descriptor_layouts[m_hash_layout_mapping[hash]];
	}

	m_descriptor_layouts.emplace_back(p_device, meta, set_index);

	m_hash_layout_mapping.emplace(hash, m_descriptor_layouts.size() - 1);
	m_descriptor_layout_table.emplace(m_descriptor_layouts.back().GetHandle(), m_descriptor_layouts.size() - 1);

	return m_descriptor_layouts[m_hash_layout_mapping[hash]];
}

VkDescriptorSet DescriptorAllocator::AllocateDescriptorSet(const ShaderReflectionData &meta, uint32_t set_index)
{
	return AllocateDescriptorSet(GetDescriptorLayout(meta, set_index));
}

VkDescriptorSet DescriptorAllocator::AllocateDescriptorSet(const VkDescriptorSetLayout &descriptor_layout)
{
	if (m_descriptor_pool_table.find(descriptor_layout) == m_descriptor_pool_table.end())
	{
		// Create new descriptor pool
		m_descriptor_pools.emplace_back(p_device, m_descriptor_layouts[m_descriptor_pool_table[descriptor_layout]]);
		m_descriptor_pool_table[descriptor_layout] = m_descriptor_pools.size() - 1;
	}

	return m_descriptor_pools[m_descriptor_pool_table[descriptor_layout]].Allocate(m_descriptor_layouts[m_descriptor_pool_table[descriptor_layout]]);
}

void DescriptorAllocator::Free(const VkDescriptorSet &descriptor_set)
{
	if (m_set_pool_mapping.find(descriptor_set) == m_set_pool_mapping.end())
	{
		return;
	}
	m_descriptor_pools[m_set_pool_mapping[descriptor_set]].Free(descriptor_set);
}
}        // namespace Ilum