#include "Command.hpp"
#include "Buffer.hpp"
#include "Device.hpp"
#include "Texture.hpp"

namespace Ilum::Vulkan
{
static std::vector<std::unordered_map<size_t, VkCommandPool>>          CommandPools;
static std::vector<std::unordered_map<size_t, std::vector<Command *>>> CommandBuffers;

static uint32_t CommandCount = 0;

Command::Command(RHIDevice *device, uint32_t frame_index, RHIQueueFamily family) :
    RHICommand(device, family)
{
	CommandCount++;

	while (CommandBuffers.size() <= frame_index)
	{
		CommandBuffers.push_back({});
		CommandPools.push_back({});
	}

	// Register Command Buffer
	size_t hash = 0;
	HashCombine(hash, frame_index, family, std::this_thread::get_id());
	CommandBuffers[frame_index][hash].push_back(this);

	// Check Command Pool
	if (CommandPools[frame_index].find(hash) == CommandPools[frame_index].end())
	{
		VkCommandPoolCreateInfo create_info = {};
		create_info.sType                   = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		create_info.flags                   = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
		create_info.queueFamilyIndex        = static_cast<Device *>(device)->GetQueueFamily(family);

		VkCommandPool pool = VK_NULL_HANDLE;
		vkCreateCommandPool(static_cast<Device *>(device)->GetDevice(), &create_info, nullptr, &pool);
		CommandPools[frame_index][hash] = pool;
	}
	m_pool = CommandPools[frame_index][hash];

	// Create Command Buffer
	VkCommandBufferAllocateInfo allocate_info = {};
	allocate_info.sType                       = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocate_info.commandPool                 = m_pool;
	allocate_info.level                       = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocate_info.commandBufferCount          = 1;
	vkAllocateCommandBuffers(static_cast<Device *>(device)->GetDevice(), &allocate_info, &m_handle);
}

Command::~Command()
{
	if (m_handle)
	{
		vkFreeCommandBuffers(static_cast<Device *>(p_device)->GetDevice(), m_pool, 1, &m_handle);
	}

	if (--CommandCount == 0)
	{
		for (auto &pool_map : CommandPools)
		{
			for (auto &[hash, pool] : pool_map)
			{
				vkDestroyCommandPool(static_cast<Device *>(p_device)->GetDevice(), pool, nullptr);
			}
		}
		CommandPools.clear();
	}

	for (auto &cmd_map : CommandBuffers)
	{
		for (auto &[hash, cmds] : cmd_map)
		{
			for (auto iter = cmds.begin(); iter != cmds.end(); iter++)
			{
				if (*iter == this)
				{
					cmds.erase(iter);
					return;
				}
			}
		}
	}
}

void Command::SetState(CommandState state)
{
	m_state = state;
}

void Command::ResetCommandPool(RHIDevice *device, uint32_t frame_index)
{
	while (CommandBuffers.size() <= frame_index)
	{
		CommandBuffers.push_back({});
		CommandPools.push_back({});
	}

	for (auto &[hash, pool] : CommandPools[frame_index])
	{
		vkResetCommandPool(static_cast<Device *>(device)->GetDevice(), pool, 0);
	}

	for (auto &[hash, cmd_buffers] : CommandBuffers[frame_index])
	{
		for (auto &cmd_buffer : cmd_buffers)
		{
			cmd_buffer->m_state = CommandState::Available;
		}
	}
}

VkCommandBuffer Command::GetHandle() const
{
	return m_handle;
}

void Command::Begin()
{
	ASSERT(m_state == CommandState::Initial);

	VkCommandBufferBeginInfo begin_info = {};
	begin_info.sType                    = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	begin_info.flags                    = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	begin_info.pInheritanceInfo         = nullptr;

	vkBeginCommandBuffer(m_handle, &begin_info);

	m_state = CommandState::Recording;
}

void Command::End()
{
	ASSERT(m_state == CommandState::Recording);
	vkEndCommandBuffer(m_handle);
	m_state = CommandState::Executable;
}

void Command::BeginPass()
{
}

void Command::EndPass()
{
}

void Command::BindVertexBuffer()
{
}

void Command::BindIndexBuffer()
{
}

void Command::BindPipeline(RHIPipelineState *pipeline_state, RHIDescriptor *descriptor)
{
}

void Command::Dispatch(uint32_t group_x, uint32_t group_y, uint32_t group_z)
{
	vkCmdDispatch(m_handle, group_x, group_y, group_z);
}

void Command::Draw(uint32_t vertex_count, uint32_t instance_count, uint32_t first_vertex, uint32_t first_instance)
{
	vkCmdDraw(m_handle, vertex_count, instance_count, first_vertex, first_instance);
}

void Command::DrawIndexed(uint32_t index_count, uint32_t instance_count, uint32_t first_index, uint32_t vertex_offset, uint32_t first_instance)
{
	vkCmdDrawIndexed(m_handle, index_count, instance_count, first_index, vertex_offset, first_instance);
}

void Command::ResourceStateTransition(const std::vector<TextureStateTransition> &texture_transitions, const std::vector<BufferStateTransition> &buffer_transitions)
{
	std::vector<VkBufferMemoryBarrier> buffer_barriers(buffer_transitions.size());
	std::vector<VkImageMemoryBarrier>  image_barriers(texture_transitions.size());

	VkPipelineStageFlags src_stage = 0;
	VkPipelineStageFlags dst_stage = 0;

	for (uint32_t i = 0; i < buffer_barriers.size(); i++)
	{
		BufferState vk_buffer_state_src = BufferState::Create(buffer_transitions[i].src);
		BufferState vk_buffer_state_dst = BufferState::Create(buffer_transitions[i].dst);
		if (vk_buffer_state_src == vk_buffer_state_dst)
		{
			continue;
		}

		buffer_barriers[i].sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
		buffer_barriers[i].srcAccessMask       = vk_buffer_state_src.access_mask;
		buffer_barriers[i].dstAccessMask       = vk_buffer_state_dst.access_mask;
		buffer_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		buffer_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		buffer_barriers[i].buffer              = static_cast<Buffer *>(buffer_transitions[i].buffer)->GetHandle();
		buffer_barriers[i].offset              = 0;
		buffer_barriers[i].size                = buffer_transitions[i].buffer->GetDesc().size;
		src_stage |= vk_buffer_state_src.stage;
		dst_stage |= vk_buffer_state_dst.stage;
	}

	for (uint32_t i = 0; i < image_barriers.size(); i++)
	{
		TextureState vk_texture_state_src = TextureState::Create(texture_transitions[i].src);
		TextureState vk_texture_state_dst = TextureState::Create(texture_transitions[i].dst);
		if (vk_texture_state_src == vk_texture_state_dst)
		{
			continue;
		}

		VkImageSubresourceRange range = {};
		range.aspectMask              = IsDepthFormat(texture_transitions[i].texture->GetDesc().format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseArrayLayer          = texture_transitions[i].range.base_layer;
		range.baseMipLevel            = texture_transitions[i].range.base_mip;
		range.layerCount              = texture_transitions[i].range.layer_count;
		range.levelCount              = texture_transitions[i].range.mip_count;

		image_barriers[i].sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		image_barriers[i].srcAccessMask       = vk_texture_state_src.access_mask;
		image_barriers[i].dstAccessMask       = vk_texture_state_dst.access_mask;
		image_barriers[i].oldLayout           = vk_texture_state_src.layout;
		image_barriers[i].newLayout           = vk_texture_state_dst.layout;
		image_barriers[i].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		image_barriers[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		image_barriers[i].image               = static_cast<Texture *>(texture_transitions[i].texture)->GetHandle();
		image_barriers[i].subresourceRange    = range;
		src_stage |= vk_texture_state_src.stage;
		dst_stage |= vk_texture_state_dst.stage;
	}

	vkCmdPipelineBarrier(m_handle, src_stage, dst_stage, 0, 0, nullptr, static_cast<uint32_t>(buffer_barriers.size()), buffer_barriers.data(), static_cast<uint32_t>(image_barriers.size()), image_barriers.data());
}
}        // namespace Ilum::Vulkan