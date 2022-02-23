#pragma once

#include "Utils/PCH.hpp"

#include "Graphics/Buffer/Buffer.h"
#include "Graphics/Image/Image.hpp"
#include "Graphics/Synchronization/Queue.hpp"
#include "Graphics/Synchronization/QueueSystem.hpp"

#include "Renderer/RenderGraph/RenderPass.hpp"

namespace Ilum
{
class LogicalDevice;
class CommandPool;
struct PassNative;

struct ImageInfo
{
	ImageReference       resource;
	VkImageUsageFlagBits usage     = VK_IMAGE_USAGE_FLAG_BITS_MAX_ENUM;
	uint32_t             mip_level = 0;
	uint32_t             layer     = 0;
};

struct BufferInfo
{
	BufferReference resource;
	uint32_t        offset = 0;
};

class CommandBuffer
{
  public:
	CommandBuffer(CommandPool &cmd_pool, QueueUsage usage = QueueUsage::Graphics, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

	CommandBuffer(QueueUsage usage = QueueUsage::Graphics, VkCommandBufferLevel level = VK_COMMAND_BUFFER_LEVEL_PRIMARY);

	~CommandBuffer();

	CommandBuffer(const CommandBuffer &) = delete;

	CommandBuffer &operator=(const CommandBuffer &) = delete;

	void reset() const;

	bool begin(VkCommandBufferUsageFlagBits usage = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, VkCommandBufferInheritanceInfo *inheritanceInfo = nullptr) const;

	bool beginRenderPass(const PassNative &pass) const;

	void endRenderPass() const;

	void end() const;

	// Copy
	void copyImage(const ImageInfo &src, const ImageInfo &dst) const;

	void copyBufferToImage(const BufferInfo &src, const ImageInfo &dst) const;

	void copyImageToBuffer(const ImageInfo &src, const BufferInfo &dst) const;

	void copyBuffer(const BufferInfo &src, const BufferInfo &dst, VkDeviceSize size) const;

	// Mipmap generate
	void blitImage(const Image &src, VkImageUsageFlagBits src_usage, const Image &dst, VkImageUsageFlagBits dst_usage, VkFilter filter) const;

	void generateMipmaps(const Image &image, VkImageUsageFlagBits initial_usage, VkFilter filter) const;

	// Transfer layout
	void transferLayout(const Image &image, VkImageUsageFlagBits old_usage, VkImageUsageFlagBits new_usage) const;

	void transferLayout(const std::vector<ImageReference> &images, VkImageUsageFlagBits old_usage, VkImageUsageFlagBits new_usage) const;

	void submitIdle();

	void submit(const VkSemaphore &wait_semaphore = VK_NULL_HANDLE, const VkSemaphore &signal_semaphore = VK_NULL_HANDLE, VkFence fence = VK_NULL_HANDLE, VkShaderStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

	void submit(const SubmitInfo &submit_info) const;

	operator const VkCommandBuffer &() const;

	const VkCommandBuffer &getCommandBuffer() const;

  private:
	CommandPool &   m_command_pool;
	VkCommandBuffer m_handle = VK_NULL_HANDLE;
	std::mutex      m_mutex;
};
}        // namespace Ilum