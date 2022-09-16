#include "Descriptor.hpp"
#include "Buffer.hpp"
#include "Definitions.hpp"
#include "Device.hpp"
#include "Sampler.hpp"
#include "Texture.hpp"

namespace Ilum::Vulkan
{
inline static VkDescriptorPool                                  DescriptorPool = VK_NULL_HANDLE;
inline static std::unordered_map<size_t, VkDescriptorSetLayout> DescriptorSetLayouts;
inline static std::unordered_map<size_t, VkDescriptorSet>       DescriptorSet;

inline static size_t DescriptorCount = 0;

inline static std::unordered_map<DescriptorType, VkDescriptorType> DescriptorTypeMap = {
    {DescriptorType::Sampler, VK_DESCRIPTOR_TYPE_SAMPLER},
    {DescriptorType::TextureSRV, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE},
    {DescriptorType::TextureUAV, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
    {DescriptorType::ConstantBuffer, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER},
    {DescriptorType::StructuredBuffer, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
    {DescriptorType::AccelerationStructure, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR},
};

Descriptor::Descriptor(RHIDevice *device, const ShaderMeta &meta) :
    RHIDescriptor(device, meta)
{
	if (DescriptorCount++ == 0)
	{
		VkPhysicalDeviceProperties properties = {};
		vkGetPhysicalDeviceProperties(static_cast<Device *>(p_device)->GetPhysicalDevice(), &properties);

		VkDescriptorPoolSize pool_sizes[] =
		    {
		        {VK_DESCRIPTOR_TYPE_SAMPLER, properties.limits.maxDescriptorSetSamplers},
		        {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, properties.limits.maxDescriptorSetSampledImages},
		        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, properties.limits.maxDescriptorSetStorageImages},
		        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, properties.limits.maxDescriptorSetUniformBuffers},
		        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, properties.limits.maxDescriptorSetStorageBuffers},
		        {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1024},
		    };

		// Create descriptor pool
		VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
		descriptor_pool_create_info.sType                      = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptor_pool_create_info.pPoolSizes                 = pool_sizes;
		descriptor_pool_create_info.poolSizeCount              = 6;
		descriptor_pool_create_info.maxSets                    = 4096;
		descriptor_pool_create_info.flags                      = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;

		vkCreateDescriptorPool(static_cast<Device *>(p_device)->GetDevice(), &descriptor_pool_create_info, nullptr, &DescriptorPool);
	}

	std::unordered_map<uint32_t, ShaderMeta> set_meta;
	for (auto &descriptor : m_meta.descriptors)
	{
		set_meta[descriptor.set].descriptors.emplace_back(descriptor);
		HashCombine(
		    set_meta[descriptor.set].hash,
		    descriptor.array_size,
		    descriptor.binding,
		    descriptor.name,
		    descriptor.set,
		    descriptor.stage,
		    descriptor.type);

		m_binding_hash[descriptor.name]      = 0;
		m_descriptor_lookup[descriptor.name] = std::make_pair(descriptor.set, descriptor.binding);

		switch (descriptor.type)
		{
			case DescriptorType::TextureSRV:
				m_texture_resolves.emplace(descriptor.name, TextureResolve{descriptor.set, descriptor.binding, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
				break;
			case DescriptorType::TextureUAV:
				m_texture_resolves.emplace(descriptor.name, TextureResolve{descriptor.set, descriptor.binding, VK_IMAGE_LAYOUT_GENERAL});
				break;
			case DescriptorType::Sampler:
				m_texture_resolves.emplace(descriptor.name, TextureResolve{descriptor.set, descriptor.binding});
				break;
			case DescriptorType::ConstantBuffer:
			case DescriptorType::StructuredBuffer:
				m_buffer_resolves.emplace(descriptor.name, BufferResolve{descriptor.set, descriptor.binding});
				break;
			case DescriptorType::AccelerationStructure:
				m_acceleration_structure_resolves.emplace(descriptor.name, AccelerationStructureResolve{descriptor.set, descriptor.binding});
				break;
			default:
				break;
		}
	}

	for (auto &constant : m_meta.constants)
	{
		if (m_constant_resolves.find(constant.name) != m_constant_resolves.end())
		{
			m_constant_resolves[constant.name].stage |= ToVulkanShaderStages(constant.stage);
		}
		else
		{
			m_constant_resolves.emplace(
			    constant.name,
			    ConstantResolve{
			        std::vector<uint8_t>(constant.size),
			        constant.offset,
			        ToVulkanShaderStages(constant.stage)});
		}
	}

	for (auto &[set, meta] : set_meta)
	{
		// Create descriptor set layout
		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (DescriptorSetLayouts.find(meta.hash) == DescriptorSetLayouts.end())
		{
			layout = CreateDescriptorSetLayout(meta);
			DescriptorSetLayouts.emplace(meta.hash, layout);
		}
		else
		{
			layout = DescriptorSetLayouts[meta.hash];
		}
		m_descriptor_set_layouts.emplace(set, layout);

		m_binding_dirty.emplace(set, false);
	}
}

Descriptor ::~Descriptor()
{
	for (auto &[set, descriptor_set] : m_descriptor_sets)
	{
		vkFreeDescriptorSets(static_cast<Device *>(p_device)->GetDevice(), DescriptorPool, 1, &descriptor_set);
	}

	m_descriptor_set_layouts.clear();
	m_descriptor_sets.clear();

	if (--DescriptorCount == 0)
	{
		if (DescriptorPool)
		{
			vkDestroyDescriptorPool(static_cast<Device *>(p_device)->GetDevice(), DescriptorPool, nullptr);
			DescriptorPool = VK_NULL_HANDLE;
		}
		for (auto &[hash, layout] : DescriptorSetLayouts)
		{
			vkDestroyDescriptorSetLayout(static_cast<Device *>(p_device)->GetDevice(), layout, nullptr);
		}
		DescriptorSetLayouts.clear();
	}
}
RHIDescriptor &Descriptor::BindTexture(const std::string &name, RHITexture *texture, RHITextureDimension dimension)
{
	TextureRange range = {};
	range.dimension    = dimension;
	range.base_layer   = 0;
	range.layer_count  = texture->GetDesc().layers;
	range.base_mip     = 0;
	range.mip_count    = texture->GetDesc().mips;

	return BindTexture(name, texture, range);
}

RHIDescriptor &Descriptor::BindTexture(const std::string &name, RHITexture *texture, const TextureRange &range)
{
	size_t hash = 0;
	HashCombine(hash, texture, range.dimension, range.base_mip, range.mip_count, range.base_layer, range.layer_count);

	if (m_binding_hash[name] != hash)
	{
		VkImageView view = static_cast<Texture *>(texture)->GetView(range);

		m_texture_resolves[name].views = {view};
		m_binding_hash[name]           = hash;

		m_binding_dirty[m_descriptor_lookup[name].first] = true;
	}

	return *this;
}

RHIDescriptor &Descriptor::BindTexture(const std::string &name, const std::vector<RHITexture *> &textures, RHITextureDimension dimension)
{
	size_t hash = 0;
	HashCombine(hash, textures, dimension);

	if (m_binding_hash[name] != hash)
	{
		m_texture_resolves[name].views.clear();
		m_texture_resolves[name].views.reserve(textures.size());

		for (auto *texture : textures)
		{
			TextureRange range = {};
			range.dimension    = dimension;
			range.base_layer   = 0;
			range.base_mip     = 0;
			range.layer_count  = texture->GetDesc().layers;
			range.mip_count    = texture->GetDesc().mips;
			VkImageView view   = static_cast<Texture *>(texture)->GetView(range);

			m_texture_resolves[name].views.push_back(view);
		}

		m_binding_hash[name] = hash;

		m_binding_dirty[m_descriptor_lookup[name].first] = true;
	}

	return *this;
}

RHIDescriptor &Descriptor::BindSampler(const std::string &name, RHISampler *sampler)
{
	size_t hash = 0;
	HashCombine(hash, sampler);

	if (m_binding_hash[name] != hash)
	{
		m_texture_resolves[name].samplers = {static_cast<Sampler *>(sampler)->GetHandle()};

		m_binding_hash[name] = hash;

		m_binding_dirty[m_descriptor_lookup[name].first] = true;
	}

	return *this;
}

RHIDescriptor &Descriptor::BindSampler(const std::string &name, const std::vector<RHISampler *> &samplers)
{
	size_t hash = 0;
	HashCombine(hash, samplers);

	if (m_binding_hash[name] != hash)
	{
		m_texture_resolves[name].samplers.clear();
		m_texture_resolves[name].samplers.reserve(samplers.size());

		for (auto *sampler : samplers)
		{
			m_texture_resolves[name].samplers.push_back(static_cast<Sampler *>(sampler)->GetHandle());
		}

		m_binding_hash[name] = hash;

		m_binding_dirty[m_descriptor_lookup[name].first] = true;
	}

	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, RHIBuffer *buffer)
{
	return BindBuffer(name, buffer, 0, buffer->GetDesc().size);
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, RHIBuffer *buffer, size_t offset, size_t range)
{
	size_t   hash          = 0;
	VkBuffer buffer_handle = static_cast<Buffer *>(buffer)->GetHandle();
	HashCombine(hash, buffer_handle, offset, range);

	if (m_binding_hash[name] != hash)
	{
		m_buffer_resolves[name].buffers = {buffer_handle};
		m_buffer_resolves[name].ranges  = {range};
		m_buffer_resolves[name].offsets = {offset};

		m_binding_hash[name] = hash;

		m_binding_dirty[m_descriptor_lookup[name].first] = true;
	}

	return *this;
}

RHIDescriptor &Descriptor::BindBuffer(const std::string &name, const std::vector<RHIBuffer *> &buffers)
{
	size_t hash = 0;
	HashCombine(hash, buffers);

	if (m_binding_hash[name] != hash)
	{
		m_buffer_resolves[name].buffers.clear();
		m_buffer_resolves[name].buffers.reserve(buffers.size());
		m_buffer_resolves[name].ranges.clear();
		m_buffer_resolves[name].ranges.reserve(buffers.size());
		m_buffer_resolves[name].offsets.clear();
		m_buffer_resolves[name].offsets.reserve(buffers.size());

		for (auto *buffer : buffers)
		{
			m_buffer_resolves[name].buffers.push_back(static_cast<Buffer *>(buffer)->GetHandle());
			m_buffer_resolves[name].ranges.push_back(buffer->GetDesc().size);
			m_buffer_resolves[name].offsets.push_back(0);
		}

		m_binding_hash[name] = hash;

		m_binding_dirty[m_descriptor_lookup[name].first] = true;
	}

	return *this;
}

RHIDescriptor &Descriptor::BindConstant(const std::string &name, const void *constant)
{
	std::memcpy(m_constant_resolves[name].data.data(), constant, m_constant_resolves[name].data.size());
	return *this;
}

RHIDescriptor &Descriptor::BindAccelerationStructure(const std::string &name, RHIAccelerationStructure *acceleration_structure)
{
	return *this;
}

const std::unordered_map<uint32_t, VkDescriptorSet> &Descriptor::GetDescriptorSet()
{
	// Check update
	for (auto &[set, dirty] : m_binding_dirty)
	{
		if (dirty)
		{
			size_t hash = 0;
			for (auto &[name, binding_hash] : m_binding_hash)
			{
				if (m_descriptor_lookup[name].first == set)
				{
					HashCombine(hash, binding_hash);
				}
			}

			if (DescriptorSet.find(hash) != DescriptorSet.end())
			{
				VkDescriptorSet descriptor_set = DescriptorSet[hash];
				m_descriptor_sets[set]         = descriptor_set;
				return m_descriptor_sets;
			}

			// Allocate descriptor set
			VkDescriptorSetAllocateInfo allocate_info = {};
			allocate_info.sType                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
			allocate_info.descriptorPool              = DescriptorPool;
			allocate_info.descriptorSetCount          = 1;
			allocate_info.pSetLayouts                 = &m_descriptor_set_layouts.at(set);

			VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
			vkAllocateDescriptorSets(static_cast<Device *>(p_device)->GetDevice(), &allocate_info, &descriptor_set);

			DescriptorSet.emplace(hash, descriptor_set);
			m_descriptor_sets[set] = descriptor_set;

			std::vector<VkWriteDescriptorSet>                write_sets;
			std::vector<std::vector<VkDescriptorImageInfo>>  image_infos  = {};
			std::vector<std::vector<VkDescriptorBufferInfo>> buffer_infos = {};
			for (auto &descriptor : m_meta.descriptors)
			{
				if (descriptor.set == set)
				{
					bool     is_texture       = false;
					bool     is_buffer        = false;
					uint32_t descriptor_count = 0;

					image_infos.push_back({});
					buffer_infos.push_back({});

					// Handle Texture
					if (descriptor.type == DescriptorType::TextureSRV ||
					    descriptor.type == DescriptorType::TextureUAV)
					{
						is_texture = true;
						for (auto &view : m_texture_resolves[descriptor.name].views)
						{
							image_infos.back().push_back(VkDescriptorImageInfo{
							    VK_NULL_HANDLE,
							    view,
							    m_texture_resolves[descriptor.name].layout});
						}
						descriptor_count = static_cast<uint32_t>(image_infos.back().size());
					}

					// Handle Sampler
					if (descriptor.type == DescriptorType::Sampler)
					{
						is_texture = true;
						for (auto &sampler : m_texture_resolves[descriptor.name].samplers)
						{
							image_infos.back().push_back(VkDescriptorImageInfo{
							    sampler,
							    VK_NULL_HANDLE,
							    VK_IMAGE_LAYOUT_UNDEFINED});
						}
						descriptor_count = static_cast<uint32_t>(image_infos.back().size());
					}

					// Handle Buffer
					if (descriptor.type == DescriptorType::ConstantBuffer ||
					    descriptor.type == DescriptorType::StructuredBuffer)
					{
						is_buffer = true;
						for (uint32_t i = 0; i < m_buffer_resolves[descriptor.name].buffers.size(); i++)
						{
							buffer_infos.back().push_back(VkDescriptorBufferInfo{
							    m_buffer_resolves[descriptor.name].buffers[i],
							    m_buffer_resolves[descriptor.name].offsets[i],
							    m_buffer_resolves[descriptor.name].ranges[i]});
						}
						descriptor_count = static_cast<uint32_t>(buffer_infos.back().size());
					}

					VkWriteDescriptorSet write_set = {};
					write_set.sType                = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
					write_set.dstSet               = m_descriptor_sets[set];
					write_set.dstBinding           = descriptor.binding;
					write_set.dstArrayElement      = 0;
					write_set.descriptorCount      = descriptor_count;
					write_set.descriptorType       = DescriptorTypeMap[descriptor.type];
					write_set.pImageInfo           = is_texture ? image_infos.back().data() : nullptr;
					write_set.pBufferInfo          = is_buffer ? buffer_infos.back().data() : nullptr;
					write_set.pTexelBufferView     = nullptr;

					write_sets.push_back(write_set);
				}
			}

			vkUpdateDescriptorSets(static_cast<Device *>(p_device)->GetDevice(), static_cast<uint32_t>(write_sets.size()), write_sets.data(), 0, nullptr);

			dirty = false;
		}
	}

	return m_descriptor_sets;
}

const std::unordered_map<uint32_t, VkDescriptorSetLayout> &Descriptor::GetDescriptorSetLayout()
{
	return m_descriptor_set_layouts;
}

const std::map<std::string, ConstantResolve> &Descriptor::GetConstantResolve() const
{
	return m_constant_resolves;
}

VkDescriptorSetLayout Descriptor::CreateDescriptorSetLayout(const ShaderMeta &meta)
{
	std::vector<VkDescriptorBindingFlags>     descriptor_binding_flags       = {};
	std::vector<VkDescriptorSetLayoutBinding> descriptor_set_layout_bindings = {};

	VkDescriptorBindingFlags binding_flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;

	for (const auto &descriptor : meta.descriptors)
	{
		VkDescriptorSetLayoutBinding layout_binding = {};
		layout_binding.binding                      = descriptor.binding;
		layout_binding.descriptorType               = DescriptorTypeMap[descriptor.type];
		layout_binding.stageFlags                   = ToVulkanShaderStage[descriptor.stage];
		layout_binding.descriptorCount              = descriptor.array_size == 0 ? 1024 : descriptor.array_size;
		descriptor_set_layout_bindings.push_back(layout_binding);
		descriptor_binding_flags.push_back(binding_flags | (descriptor.array_size == 0 ? VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0));
	}

	VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info = {};
	descriptor_set_layout_create_info.sType                           = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
	descriptor_set_layout_create_info.bindingCount                    = static_cast<uint32_t>(descriptor_set_layout_bindings.size());
	descriptor_set_layout_create_info.pBindings                       = descriptor_set_layout_bindings.data();
	descriptor_set_layout_create_info.flags                           = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;

	VkDescriptorSetLayoutBindingFlagsCreateInfo descriptor_set_layout_binding_flag_create_info = {};
	descriptor_set_layout_binding_flag_create_info.sType                                       = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
	descriptor_set_layout_binding_flag_create_info.bindingCount                                = static_cast<uint32_t>(descriptor_binding_flags.size());
	descriptor_set_layout_binding_flag_create_info.pBindingFlags                               = descriptor_binding_flags.data();

	descriptor_set_layout_create_info.pNext = &descriptor_set_layout_binding_flag_create_info;

	VkDescriptorSetLayout layout = VK_NULL_HANDLE;
	vkCreateDescriptorSetLayout(static_cast<Device *>(p_device)->GetDevice(), &descriptor_set_layout_create_info, nullptr, &layout);

	return layout;
}
}        // namespace Ilum::Vulkan