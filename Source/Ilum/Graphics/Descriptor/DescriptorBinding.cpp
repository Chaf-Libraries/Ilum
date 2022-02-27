#include "DescriptorBinding.hpp"

#include "Device/LogicalDevice.hpp"

#include "Graphics/GraphicsContext.hpp"

namespace Ilum
{
inline VkImageUsageFlagBits type_to_image_usage(VkDescriptorType type)
{
	switch (type)
	{
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
			return VK_IMAGE_USAGE_SAMPLED_BIT;
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
			return VK_IMAGE_USAGE_SAMPLED_BIT;
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
			return VK_IMAGE_USAGE_STORAGE_BIT;
		case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
			return VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
		default:
			return VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM;
	}
}

inline VkBufferUsageFlagBits type_to_buffer_usage(VkDescriptorType type)
{
	switch (type)
	{
		case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
			return VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
			return VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
			return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
			return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
			return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
			return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
			return VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
		case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR:
			return VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
		default:
			return VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM;
	}
}

inline bool is_buffer(VkDescriptorType type)
{
	switch (type)
	{
		case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
		case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
		case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
		case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT:
			return true;
		default:
			return false;
	}
}

inline bool is_image(VkDescriptorType type)
{
	switch (type)
	{
		case VK_DESCRIPTOR_TYPE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
		case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
		case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
			return true;
		default:
			return false;
	}
}

size_t DescriptorBinding::allocate(uint32_t set, const Buffer &buffer, VkDescriptorType type)
{
	m_buffer_writes[set].push_back(BufferWriteInfo{std::addressof(buffer), type_to_buffer_usage(type)});
	return m_buffer_writes[set].size() - 1;
}

size_t DescriptorBinding::allocate(uint32_t set, const Image &image, ImageViewType view, VkDescriptorType type)
{
	m_image_writes[set].push_back(ImageWriteInfo{std::addressof(image), type_to_image_usage(type), view, {}});
	return m_image_writes[set].size() - 1;
}

size_t DescriptorBinding::allocate(uint32_t set, const Image &image, const Sampler &sampler, ImageViewType view, VkDescriptorType type)
{
	m_image_writes[set].push_back(ImageWriteInfo{std::addressof(image), type_to_image_usage(type), view, std::addressof(sampler)});
	return m_image_writes[set].size() - 1;
}

size_t DescriptorBinding::allocate(uint32_t set, const Sampler &sampler)
{
	m_image_writes[set].push_back(ImageWriteInfo{{}, VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM, {}, std::addressof(sampler)});
	return m_image_writes[set].size() - 1;
}

size_t DescriptorBinding::allocate(uint32_t set, const AccelerationStructure &acceleration_structure)
{
	m_acceleration_structure_writes[set].push_back(AccelerationStructureWriteInfo{std::addressof(acceleration_structure)});
	return m_acceleration_structure_writes[set].size() - 1;
}

DescriptorBinding &DescriptorBinding::bind(uint32_t set, uint32_t binding, const std::string &name, VkDescriptorType type)
{
	if (type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
	{
		m_acceleration_structure_to_resolves[set].push_back(AccelerationStructureToResolve{name, binding});
		return *this;
	}

	if (type_to_buffer_usage(type) == VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM)
	{
		return bind(set, binding, name, Sampler(), ImageViewType::Native, type);
	}

	m_buffer_to_resolves[set].push_back(BufferToResolve{name, binding, type, type_to_buffer_usage(type)});
	return *this;
}

DescriptorBinding &DescriptorBinding::bind(uint32_t set, uint32_t binding, const std::string &name, ImageViewType view, VkDescriptorType type)
{
	return bind(set, binding, name, Sampler(), view, type);
}

DescriptorBinding &DescriptorBinding::bind(uint32_t set, uint32_t binding, const std::string &name, const Sampler &sampler, VkDescriptorType type)
{
	m_sampler_to_resolves[set].push_back(SamplerToResolve{std::addressof(sampler), binding, type});
	return *this;
}

DescriptorBinding &DescriptorBinding::bind(uint32_t set, uint32_t binding, const std::string &name, const Sampler &sampler, ImageViewType view, VkDescriptorType type)
{
	m_image_to_resolves[set].push_back(ImageToResolve{name, binding, type, type_to_image_usage(type), view, std::addressof(sampler)});
	return *this;
}

DescriptorBinding &DescriptorBinding::bind(uint32_t set, uint32_t binding, const Sampler &sampler, VkDescriptorType type)
{
	m_sampler_to_resolves[set].push_back(SamplerToResolve{std::addressof(sampler), binding, type});
	return *this;
}

void DescriptorBinding::setOption(ResolveOption option)
{
	m_options = option;
}

ResolveOption DescriptorBinding::getOption() const
{
	return m_options;
}

void DescriptorBinding::resolve(const ResolveInfo &resolve_info)
{
	m_buffer_writes.clear();
	m_image_writes.clear();
	m_descriptor_writes.clear();
	m_acceleration_structure_writes.clear();

	// Resolve images
	for (const auto &[set, image_to_resolves] : m_image_to_resolves)
	{
		for (const auto &image_to_resolve : image_to_resolves)
		{
			if (resolve_info.getImages().find(image_to_resolve.name) == resolve_info.getImages().end())
			{
				continue;
			}

			auto   images = resolve_info.getImages().at(image_to_resolve.name);
			size_t index  = 0;
			if (image_to_resolve.sampler_handle->getSampler())
			{
				for (const auto &image : images)
				{
					index = allocate(set, image.get(), *image_to_resolve.sampler_handle, image_to_resolve.view, image_to_resolve.type);
				}
			}
			else
			{
				for (const auto &image : images)
				{
					index = allocate(set, image.get(), image_to_resolve.view, image_to_resolve.type);
				}
			}

			m_descriptor_writes[set].push_back(DescriptorWriteInfo{
			    image_to_resolve.type,
			    image_to_resolve.binding,
			    static_cast<uint32_t>(index + 1 - images.size()),
			    static_cast<uint32_t>(images.size())});
		}
	}

	// Resolve buffers
	for (const auto &[set, buffer_to_resolves] : m_buffer_to_resolves)
	{
		for (const auto &buffer_to_resolve : buffer_to_resolves)
		{
			auto & buffers = resolve_info.getBuffers().at(buffer_to_resolve.name);
			size_t index   = 0;
			for (const auto &buffer : buffers)
			{
				index = allocate(set, buffer.get(), buffer_to_resolve.type);
			}
			m_descriptor_writes[set].push_back(DescriptorWriteInfo{
			    buffer_to_resolve.type,
			    buffer_to_resolve.binding,
			    static_cast<uint32_t>(index + 1 - buffers.size()),
			    static_cast<uint32_t>(buffers.size())});
		}
	}

	// Resolve samplers
	for (const auto &[set, sampler_to_resolves] : m_sampler_to_resolves)
	{
		for (const auto &sampler_to_resolve : sampler_to_resolves)
		{
			size_t index = allocate(set, *sampler_to_resolve.sampler_handle);
			m_descriptor_writes[set].push_back(DescriptorWriteInfo{
			    sampler_to_resolve.type,
			    sampler_to_resolve.binding,
			    static_cast<uint32_t>(index),
			    1});
		}
	}

	// Resolve acceleration structures
	for (const auto &[set, acceleration_structure_to_resolves] : m_acceleration_structure_to_resolves)
	{
		for (const auto &acceleration_structure_to_resolve : acceleration_structure_to_resolves)
		{
			auto & acceleration_structures = resolve_info.getAccelerationStructures().at(acceleration_structure_to_resolve.name);
			size_t index                   = 0;
			for (const auto &acceleration_structure : acceleration_structures)
			{
				index = allocate(set, acceleration_structure.get());
			}

			m_descriptor_writes[set].push_back(DescriptorWriteInfo{
			    VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			    acceleration_structure_to_resolve.binding,
			    static_cast<uint32_t>(index + 1 - acceleration_structures.size()),
			    static_cast<uint32_t>(acceleration_structures.size())});
		}
	}
}

void DescriptorBinding::write(const DescriptorSet &descriptor_set)
{
	if (m_options == ResolveOption::None)
	{
		return;
	}
	if (m_options == ResolveOption::Once)
	{
		m_options = ResolveOption::None;
	}

	GraphicsContext::instance()->getQueueSystem().waitAll();

	std::vector<VkWriteDescriptorSet>                         write_descriptor_sets;
	std::vector<VkDescriptorBufferInfo>                       descriptor_buffer_infos;
	std::vector<VkDescriptorImageInfo>                        descriptor_image_infos;
	std::vector<VkWriteDescriptorSetAccelerationStructureKHR> descriptor_as_infos;

	write_descriptor_sets.reserve(m_descriptor_writes.size());
	descriptor_buffer_infos.reserve(m_buffer_to_resolves.size());
	descriptor_image_infos.reserve(m_image_to_resolves.size());
	descriptor_as_infos.reserve(m_acceleration_structure_to_resolves.size());

	// Write buffers
	for (const auto &buffer_info : m_buffer_writes[descriptor_set.index()])
	{
		descriptor_buffer_infos.push_back(VkDescriptorBufferInfo{
		    buffer_info.handle->getBuffer(),
		    0,
		    buffer_info.handle->getSize()});
	}

	// Write images
	for (const auto &image_info : m_image_writes[descriptor_set.index()])
	{
		descriptor_image_infos.push_back(VkDescriptorImageInfo{
		    image_info.sampler_handle != nullptr ? image_info.sampler_handle->getSampler() : VK_NULL_HANDLE,
		    image_info.handle != nullptr ? image_info.handle->getView(image_info.view) : VK_NULL_HANDLE,
		    Image::usage_to_layout(image_info.usage)});
	}

	// Write acceleration structures
	for (const auto &acceleration_structure_info : m_acceleration_structure_writes[descriptor_set.index()])
	{
		VkWriteDescriptorSetAccelerationStructureKHR write_descriptor_set_as = {};

		write_descriptor_set_as.sType                      = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
		write_descriptor_set_as.accelerationStructureCount = 1;
		write_descriptor_set_as.pAccelerationStructures    = &acceleration_structure_info.handle->getHandle();

		descriptor_as_infos.push_back(write_descriptor_set_as);
	}

	// Write descriptor
	for (const auto &write : m_descriptor_writes[descriptor_set.index()])
	{
		write_descriptor_sets.push_back(VkWriteDescriptorSet{
		    /*sType*/ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		    /*pNext*/ write.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR ? descriptor_as_infos.data() + write.first_index : nullptr,
		    /*dstSet*/ descriptor_set,
		    /*dstBinding*/ write.binding,
		    /*dstArrayElement*/ 0,
		    /*descriptorCount*/ write.count,
		    /*descriptorType*/ write.type,
		    /*pImageInfo*/ is_image(write.type) ? descriptor_image_infos.data() + write.first_index : nullptr,
		    /*pBufferInfo*/ is_buffer(write.type) ? descriptor_buffer_infos.data() + write.first_index : nullptr,
		    /*pTexelBufferView*/ nullptr});
	}

	descriptor_set.update(write_descriptor_sets);
}

void DescriptorBinding::write(const std::vector<DescriptorSet> &descriptor_sets)
{
	for (auto &descriptor_set : descriptor_sets)
	{
		write(descriptor_set);
	}
}

const std::map<uint32_t, std::vector<DescriptorBinding::BufferToResolve>> &DescriptorBinding::getBoundBuffers() const
{
	return m_buffer_to_resolves;
}

const std::map<uint32_t, std::vector<DescriptorBinding::ImageToResolve>> &DescriptorBinding::getBoundImages() const
{
	return m_image_to_resolves;
}

const std::map<uint32_t, std::vector<DescriptorBinding::AccelerationStructureToResolve>> &DescriptorBinding::getAccelerationStructures() const
{
	return m_acceleration_structure_to_resolves;
}

void ResolveInfo::resolve(const std::string &name, const Buffer &buffer)
{
	m_buffer_resolves[name] = {buffer};
}

void ResolveInfo::resolve(const std::string &name, const Image &image)
{
	m_image_resolves[name] = {image};
}

void ResolveInfo::resolve(const std::string &name, const AccelerationStructure &acceleration_structure)
{
	m_acceleration_structure_resolves[name] = {acceleration_structure};
}

void ResolveInfo::resolve(const std::string &name, const std::vector<BufferReference> &buffers)
{
	for (const auto &buffer : buffers)
	{
		m_buffer_resolves[name].push_back(buffer);
	}
}

void ResolveInfo::resolve(const std::string &name, const std::vector<ImageReference> &images)
{
	for (const auto &image : images)
	{
		m_image_resolves[name].push_back(image);
	}
}

void ResolveInfo::resolve(const std::string &name, const std::vector<AccelerationStructureReference> &acceleration_structures)
{
	for (const auto &acceleration_structure : acceleration_structures)
	{
		m_acceleration_structure_resolves[name].push_back(acceleration_structure);
	}
}

const std::unordered_map<std::string, std::vector<BufferReference>> &ResolveInfo::getBuffers() const
{
	return m_buffer_resolves;
}

const std::unordered_map<std::string, std::vector<ImageReference>> &ResolveInfo::getImages() const
{
	return m_image_resolves;
}

const std::unordered_map<std::string, std::vector<AccelerationStructureReference>> &ResolveInfo::getAccelerationStructures() const
{
	return m_acceleration_structure_resolves;
}
}        // namespace Ilum