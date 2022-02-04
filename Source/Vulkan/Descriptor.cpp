#include "Descriptor.hpp"
#include "Device.hpp"
#include "RenderContext.hpp"
#include "Shader.hpp"

#include <Core/Hash.hpp>

namespace Ilum::Vulkan
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

DescriptorSetLayout::DescriptorSetLayout(const ReflectionData &reflection_data, uint32_t set) :
    m_set(set)
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

	vkCreateDescriptorSetLayout(RenderContext::GetDevice(), &descriptor_set_layout_create_info, nullptr, &m_handle);
}

DescriptorSetLayout::~DescriptorSetLayout()
{
	if (m_handle)
	{
		vkDestroyDescriptorSetLayout(RenderContext::GetDevice(), m_handle, nullptr);
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

DescriptorPool::DescriptorPool(const DescriptorSetLayout &descriptor_layout, uint32_t pool_size) :
    m_pool_max_sets(pool_size), m_descriptor_layout(descriptor_layout)
{
	auto &bindings = descriptor_layout.GetBinding();

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
}

DescriptorPool::~DescriptorPool()
{
	for (auto &pool : m_descriptor_pools)
	{
		vkDestroyDescriptorPool(RenderContext::GetDevice(), pool, nullptr);
	}
}

void DescriptorPool::Reset()
{
	for (auto &pool : m_descriptor_pools)
	{
		vkResetDescriptorPool(RenderContext::GetDevice(), pool, 0);
	}

	std::fill(m_pool_sets_count.begin(), m_pool_sets_count.end(), 0);
	m_set_pool_mapping.clear();

	m_pool_index = 0;
}

bool DescriptorPool::Has(VkDescriptorSet descriptor_set)
{
	return m_set_pool_mapping.find(descriptor_set) != m_set_pool_mapping.end();
}

VkDescriptorSet DescriptorPool::Allocate()
{
	m_pool_index = FindAvaliablePool(m_pool_index);

	++m_pool_sets_count[m_pool_index];

	VkDescriptorSetAllocateInfo allocate_info = {};
	allocate_info.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
	allocate_info.descriptorPool              = m_descriptor_pools[m_pool_index];
	allocate_info.descriptorSetCount          = 1;
	allocate_info.pSetLayouts                 = &m_descriptor_layout.GetHandle();

	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

	if (!VK_CHECK(vkAllocateDescriptorSets(RenderContext::GetDevice(), &allocate_info, &descriptor_set)))
	{
		LOG_ERROR("Failed to allocate descriptor set!");

		--m_pool_sets_count[m_pool_index];
		return VK_NULL_HANDLE;
	}

	m_set_pool_mapping.emplace(descriptor_set, m_pool_index);

	return descriptor_set;
}

void DescriptorPool::Free(const VkDescriptorSet &descriptor_set)
{
	auto it = m_set_pool_mapping.find(descriptor_set);

	if (it == m_set_pool_mapping.end())
	{
		return;
	}

	auto desc_pool_index = it->second;

	vkFreeDescriptorSets(RenderContext::GetDevice(), m_descriptor_pools[desc_pool_index], 1, &descriptor_set);

	m_set_pool_mapping.erase(it);

	--m_pool_sets_count[desc_pool_index];

	m_pool_index = desc_pool_index;
}

uint32_t DescriptorPool::FindAvaliablePool(uint32_t pool_index)
{
	// Create a new pool
	if (m_descriptor_pools.size() <= pool_index)
	{
		VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
		descriptor_pool_create_info.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptor_pool_create_info.pPoolSizes                 = m_pool_sizes.data();
		descriptor_pool_create_info.poolSizeCount              = static_cast<uint32_t>(m_pool_sizes.size());
		descriptor_pool_create_info.maxSets                    = m_pool_max_sets;

		VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

		if (!VK_CHECK(vkCreateDescriptorPool(RenderContext::GetDevice(), &descriptor_pool_create_info, nullptr, &descriptor_pool)))
		{
			LOG_ERROR("Failed to create descriptor pool!");
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

	return FindAvaliablePool(++pool_index);
}

const DescriptorSetLayout &DescriptorCache::RequestDescriptorSetLayout(const ReflectionData &reflection_data, uint32_t set, const std::string &debug_name)
{
	size_t hash = 0;
	Core::HashCombine(hash, reflection_data.hash);
	Core::HashCombine(hash, set);

	if (m_descriptor_layouts.find(hash) != m_descriptor_layouts.end())
	{
		return *m_descriptor_layouts[hash];
	}

	m_descriptor_layouts.emplace(hash, std::make_unique<DescriptorSetLayout>(reflection_data, set));

	if (!debug_name.empty())
	{
		VKDebugger::SetName(*m_descriptor_layouts[hash], debug_name.c_str());
	}

	return *m_descriptor_layouts[hash];
}

VkDescriptorSet DescriptorCache::RequestDescriptorSet(const ReflectionData &reflection_data, uint32_t set, const std::string &debug_name)
{
	auto &pool           = RequestDescriptorPool(reflection_data, set);
	auto  descriptor_set = pool.Allocate();

	if (!debug_name.empty())
	{
		VKDebugger::SetName(descriptor_set, debug_name.c_str());
	}

	return descriptor_set;
}

void DescriptorCache::Free(const VkDescriptorSet &descriptor_set)
{
	for (auto &[hash, pool] : m_descriptor_pools)
	{
		if (pool->Has(descriptor_set))
		{
			pool->Free(descriptor_set);
			return;
		}
	}
}

DescriptorPool &DescriptorCache::RequestDescriptorPool(const ReflectionData &reflection_data, uint32_t set, const std::string &debug_name)
{
	size_t hash = 0;
	Core::HashCombine(hash, reflection_data.hash);
	Core::HashCombine(hash, set);

	if (m_descriptor_pools.find(hash) != m_descriptor_pools.end())
	{
		return *m_descriptor_pools[hash];
	}

	m_descriptor_pools.emplace(hash, std::make_unique<DescriptorPool>(RequestDescriptorSetLayout(reflection_data, set)));

	return *m_descriptor_pools[hash];
}
}        // namespace Ilum::Vulkan