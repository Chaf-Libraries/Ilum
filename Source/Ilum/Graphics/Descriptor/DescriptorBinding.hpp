#pragma once

#include "Utils/PCH.hpp"

#include "Graphics/Buffer/Buffer.h"
#include "Graphics/Descriptor/DescriptorSet.hpp"
#include "Graphics/Image/Image.hpp"
#include "Graphics/Image/Sampler.hpp"

namespace Ilum
{
class ResolveInfo
{
  public:
	void resolve(const std::string &name, const Buffer &buffer);

	void resolve(const std::string &name, const Image &image);

	void resolve(const std::string &name, const std::vector<BufferReference> &buffers);

	void resolve(const std::string &name, const std::vector<ImageReference> &images);

	const std::unordered_map<std::string, std::vector<BufferReference>> &getBuffers() const;

	const std::unordered_map<std::string, std::vector<ImageReference>> &getImages() const;

  private:
	std::unordered_map<std::string, std::vector<BufferReference>> m_buffer_resolves;
	std::unordered_map<std::string, std::vector<ImageReference>>  m_image_resolves;
};

enum class ResolveOption
{
	Each_Frame,
	Once,
	None
};

class DescriptorBinding
{
  public:
  private:
	struct DescriptorWriteInfo
	{
		VkDescriptorType type;
		uint32_t         binding;
		uint32_t         first_index;
		uint32_t         count;
	};

	struct BufferWriteInfo
	{
		const Buffer *        handle;
		VkBufferUsageFlagBits usage;
	};

	struct ImageWriteInfo
	{
		const Image *        handle;
		VkImageUsageFlagBits usage;
		ImageViewType        view;
		const Sampler *      sampler_handle;
	};

	struct ImageToResolve
	{
		std::string          name;
		uint32_t             binding;
		VkDescriptorType     type;
		VkImageUsageFlagBits usage;
		ImageViewType        view;
		const Sampler *      sampler_handle;
	};

	struct SamplerToResolve
	{
		const Sampler *  sampler_handle;
		uint32_t         binding;
		VkDescriptorType type;
	};

	struct BufferToResolve
	{
		std::string           name;
		uint32_t              binding;
		VkDescriptorType      type;
		VkBufferUsageFlagBits usage;
	};

	std::map<uint32_t, std::vector<DescriptorWriteInfo>> m_descriptor_writes;
	std::map<uint32_t, std::vector<BufferWriteInfo>>     m_buffer_writes;
	std::map<uint32_t, std::vector<ImageWriteInfo>>      m_image_writes;
	std::map<uint32_t, std::vector<ImageToResolve>>      m_image_to_resolves;
	std::map<uint32_t, std::vector<SamplerToResolve>>    m_sampler_to_resolves;
	std::map<uint32_t, std::vector<BufferToResolve>>     m_buffer_to_resolves;

	ResolveOption m_options = ResolveOption::Each_Frame;

	size_t allocate(uint32_t set, const Buffer &buffer, VkDescriptorType type);
	size_t allocate(uint32_t set, const Image &image, ImageViewType view, VkDescriptorType type);
	size_t allocate(uint32_t set, const Image &image, const Sampler &sampler, ImageViewType view, VkDescriptorType type);
	size_t allocate(uint32_t set, const Sampler &sampler);

  public:
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const std::string &name, VkDescriptorType type);
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const std::string &name, ImageViewType view, VkDescriptorType type);
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const std::string &name, const Sampler &sampler, VkDescriptorType type);
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const std::string &name, const Sampler &sampler, ImageViewType view, VkDescriptorType type);
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const Sampler &sampler, VkDescriptorType type);

	void setOption(ResolveOption option);

	void resolve(const ResolveInfo &resolve_info);

	void write(const DescriptorSet &descriptor_set);

	void write(const std::vector<DescriptorSet> &descriptor_sets);

	const std::map<uint32_t, std::vector<BufferToResolve>> &getBoundBuffers() const;

	const std::map<uint32_t, std::vector<ImageToResolve>> &getBoundImages() const;
};
}        // namespace Ilum