#pragma once

#include "Graphics/Vulkan.hpp"

#include "Resource/Buffer.hpp"
#include "Resource/Image.hpp"

namespace Ilum::Graphics
{
class CommandPool;
class Device;

class CommandBuffer
{
  public:
	enum class State
	{
		Initial,
		Recording,
		Executable,
		Invalid
	};

  public:
	CommandBuffer(const Device &device, const CommandPool &cmd_pool, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);
	~CommandBuffer();

	CommandBuffer(const CommandBuffer &) = delete;
	CommandBuffer &operator=(const CommandBuffer &) = delete;
	CommandBuffer(CommandBuffer &&)                 = delete;
	CommandBuffer &operator=(CommandBuffer &&) = delete;

	operator const VkCommandBuffer &() const;

	const VkCommandBuffer &GetHandle() const;
	const State &          GetState() const;
	VkCommandBufferLevel   GetLevel() const;

	// Command buffer recording
	void Reset();
	void Begin(VkCommandBufferUsageFlagBits usage = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VkCommandBufferInheritanceInfo *inheritanceInfo = nullptr);
	void End();
	void SubmitIdle(uint32_t queue_index = 0);

	// Copy image and buffer
	void CopyImage(const ImageInfo &src, const ImageInfo &dst) const;
	void CopyBuffer(const BufferInfo &src, const BufferInfo &dst, VkDeviceSize size) const;
	void CopyBufferToImage(const BufferInfo &src, const ImageInfo &dst) const;
	void CopyImageToBuffer(const ImageInfo &src, const BufferInfo &dst) const;

	// Image blit and mipmap
	void BlitImage(const Image &src, VkImageUsageFlagBits src_usage, const Image &dst, VkImageUsageFlagBits dst_usage, VkFilter filter) const;
	void GenerateMipmap(const Image &image, VkImageUsageFlagBits initial_usage, VkFilter filter) const;

	// Layout transfer
	void TransferLayout(const Image &image, VkImageUsageFlagBits src_usage, VkImageUsageFlagBits dst_usage) const;
	void TransferLayout(const std::vector<ImageReference> &images, VkImageUsageFlagBits src_usage, VkImageUsageFlagBits dst_usage) const;

  private:
	const Device &     m_device;
	const CommandPool &m_pool;

	VkCommandBuffer m_handle = VK_NULL_HANDLE;

	VkCommandBufferLevel m_level;
	State                m_state = State::Invalid;
};
}        // namespace Ilum::Graphics