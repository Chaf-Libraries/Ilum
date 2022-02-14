#pragma once

#include "Utils/PCH.hpp"

#include <Graphics/Resource/Buffer.hpp>
#include "Graphics/Descriptor/DescriptorSet.hpp"
#include <Graphics/Resource/Image.hpp>
#include <Graphics/Resource/Sampler.hpp>

namespace Ilum
{
class ResolveInfo
{
  public:
	void resolve(const std::string &name, const Graphics::Buffer &buffer);

	void resolve(const std::string &name, const Graphics::Image &image);

	void resolve(const std::string &name, const std::vector<Graphics::BufferReference> &buffers);

	void resolve(const std::string &name, const std::vector<Graphics::ImageReference> &images);

	const std::unordered_map<std::string, std::vector<Graphics::BufferReference>> &getBuffers() const;

	const std::unordered_map<std::string, std::vector<Graphics::ImageReference>> &getImages() const;

  private:
	std::unordered_map<std::string, std::vector<Graphics::BufferReference>> m_buffer_resolves;
	std::unordered_map<std::string, std::vector<Graphics::ImageReference>>  m_image_resolves;
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
	struct DescriptorWriteInfo
	{
		VkDescriptorType type;
		uint32_t         binding;
		uint32_t         first_index;
		uint32_t         count;
	};

	struct BufferWriteInfo
	{
		const Graphics::Buffer *        handle;
		VkBufferUsageFlagBits usage;
	};

	struct ImageWriteInfo
	{
		const Graphics::Image *        handle;
		VkImageUsageFlagBits usage;
		Graphics::ImageViewType        view;
		const Graphics::Sampler *      sampler_handle;
	};

	struct ImageToResolve
	{
		std::string          name;
		uint32_t             binding;
		VkDescriptorType     type;
		VkImageUsageFlagBits usage;
		Graphics::ImageViewType view;
		const Graphics::Sampler *sampler_handle;
	};

	struct SamplerToResolve
	{
		const Graphics::Sampler *sampler_handle;
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

  private:
	std::map<uint32_t, std::vector<DescriptorWriteInfo>> m_descriptor_writes;
	std::map<uint32_t, std::vector<BufferWriteInfo>>     m_buffer_writes;
	std::map<uint32_t, std::vector<ImageWriteInfo>>      m_image_writes;
	std::map<uint32_t, std::vector<ImageToResolve>>      m_image_to_resolves;
	std::map<uint32_t, std::vector<SamplerToResolve>>    m_sampler_to_resolves;
	std::map<uint32_t, std::vector<BufferToResolve>>     m_buffer_to_resolves;

	ResolveOption m_options = ResolveOption::Once;

	size_t allocate(uint32_t set, const Graphics::Buffer &buffer, VkDescriptorType type);
	size_t allocate(uint32_t set, const Graphics::Image &image, Graphics::ImageViewType view, VkDescriptorType type);
	size_t allocate(uint32_t set, const Graphics::Image &image, const Graphics::Sampler &sampler, Graphics::ImageViewType view, VkDescriptorType type);
	size_t allocate(uint32_t set, const Graphics::Sampler &sampler);

  public:
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const std::string &name, VkDescriptorType type);
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const std::string &name, Graphics::ImageViewType view, VkDescriptorType type);
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const std::string &name, const Graphics::Sampler &sampler, VkDescriptorType type);
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const std::string &name, const Graphics::Sampler &sampler, Graphics::ImageViewType view, VkDescriptorType type);
	DescriptorBinding &bind(uint32_t set, uint32_t binding, const Graphics::Sampler &sampler, VkDescriptorType type);

	void setOption(ResolveOption option);

	ResolveOption getOption() const;

	void resolve(const ResolveInfo &resolve_info);

	void write(const DescriptorSet &descriptor_set);

	void write(const std::vector<DescriptorSet> &descriptor_sets);

	const std::map<uint32_t, std::vector<BufferToResolve>> &getBoundBuffers() const;

	const std::map<uint32_t, std::vector<ImageToResolve>> &getBoundImages() const;
};
}        // namespace Ilum