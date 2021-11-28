#include "HizPass.hpp"

#include "Renderer/Renderer.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Vulkan/VK_Debugger.h"

#include "Device/LogicalDevice.hpp"

namespace Ilum::pass
{
HizPass::HizPass()
{
	m_views.resize(Renderer::instance()->Last_Frame.hiz_buffer->getMipLevelCount());

	VkImageViewCreateInfo image_view_create_info       = {};
	image_view_create_info.sType                       = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
	image_view_create_info.viewType                    = VK_IMAGE_VIEW_TYPE_2D;
	image_view_create_info.format                      = VK_FORMAT_R32_SFLOAT;
	image_view_create_info.components                  = {VK_COMPONENT_SWIZZLE_R};
	image_view_create_info.subresourceRange.layerCount = 1;
	image_view_create_info.image                       = *Renderer::instance()->Last_Frame.hiz_buffer;

	for (uint32_t i = 0; i < Renderer::instance()->Last_Frame.hiz_buffer->getMipLevelCount(); i++)
	{
		image_view_create_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
		image_view_create_info.subresourceRange.baseArrayLayer = 0;
		image_view_create_info.subresourceRange.layerCount     = 1;
		image_view_create_info.subresourceRange.baseMipLevel   = i;
		image_view_create_info.subresourceRange.levelCount     = 1;

		vkCreateImageView(GraphicsContext::instance()->getLogicalDevice(), &image_view_create_info, nullptr, &m_views[i]);
	}
}

HizPass::~HizPass()
{
	for (auto &view : m_views)
	{
		vkDestroyImageView(GraphicsContext::instance()->getLogicalDevice(), view, nullptr);
	}
}

void HizPass::setupPipeline(PipelineState &state)
{
	state.shader.load(std::string(PROJECT_SOURCE_DIR) + "Asset/Shader/GLSL/hiz.comp", VK_SHADER_STAGE_COMPUTE_BIT, Shader::Type::GLSL);

	for (uint32_t level = 0; level < m_views.size(); level++)
	{
		m_descriptor_sets.push_back(DescriptorSet(state.shader));
	}

	for (uint32_t level = 0; level < m_views.size(); level++)
	{
		VkDescriptorImageInfo dstTarget;
		dstTarget.sampler     = Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp);
		dstTarget.imageView   = m_views[level];
		dstTarget.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

		VkDescriptorImageInfo srcTarget;
		srcTarget.sampler = Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp);
		if (level == 0)
		{
			srcTarget.imageView   = Renderer::instance()->Last_Frame.depth_buffer->getView();
			srcTarget.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		}
		else
		{
			srcTarget.imageView   = m_views[level - 1];
			srcTarget.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
		}

		std::vector<VkWriteDescriptorSet> write_descriptor_sets(2);
		write_descriptor_sets[0]                  = {};
		write_descriptor_sets[0].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_descriptor_sets[0].dstSet           = m_descriptor_sets[level];
		write_descriptor_sets[0].descriptorType   = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		write_descriptor_sets[0].dstBinding       = 0;
		write_descriptor_sets[0].pImageInfo       = &srcTarget;
		write_descriptor_sets[0].pBufferInfo      = nullptr;
		write_descriptor_sets[0].pTexelBufferView = nullptr;
		write_descriptor_sets[0].descriptorCount  = 1;
		write_descriptor_sets[0].pNext            = nullptr;

		write_descriptor_sets[1]                  = {};
		write_descriptor_sets[1].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_descriptor_sets[1].dstSet           = m_descriptor_sets[level];
		write_descriptor_sets[1].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
		write_descriptor_sets[1].dstBinding       = 1;
		write_descriptor_sets[1].pImageInfo       = &dstTarget;
		write_descriptor_sets[1].pBufferInfo      = nullptr;
		write_descriptor_sets[1].pTexelBufferView = nullptr;
		write_descriptor_sets[1].descriptorCount  = 1;
		write_descriptor_sets[1].pNext            = nullptr;

		m_descriptor_sets[level].update(write_descriptor_sets);
	}

	state.descriptor_bindings.bind(0, 0, "depth - buffer", Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp), ImageViewType::Native, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
	state.descriptor_bindings.bind(0, 1, "hiz - buffer", ImageViewType::Native, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void HizPass::resolveResources(ResolveState &resolve)
{
	resolve.resolve("depth - buffer", *Renderer::instance()->Last_Frame.depth_buffer);
	resolve.resolve("hiz - buffer", *Renderer::instance()->Last_Frame.hiz_buffer);
}

void HizPass::render(RenderPassState &state)
{
	auto &cmd_buffer = state.command_buffer;

	vkCmdBindPipeline(cmd_buffer, state.pass.bind_point, state.pass.pipeline);

	for (uint32_t level = 0; level < m_views.size(); level++)
	{
		{
			VkImageMemoryBarrier read_to_write{};
			read_to_write.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			read_to_write.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
			read_to_write.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
			read_to_write.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
			read_to_write.dstAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
			read_to_write.image               = *Renderer::instance()->Last_Frame.hiz_buffer;
			read_to_write.subresourceRange    = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1};
			read_to_write.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			read_to_write.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			vkCmdPipelineBarrier(
			    cmd_buffer,
			    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			    VK_DEPENDENCY_BY_REGION_BIT,
			    0, nullptr,
			    0, nullptr,
			    1, &read_to_write);
		}

		vkCmdBindDescriptorSets(cmd_buffer, state.pass.bind_point, state.pass.pipeline_layout, 0, 1, &m_descriptor_sets[level].getDescriptorSet(), 0, nullptr);

		VkExtent2D extent = {Renderer::instance()->Last_Frame.hiz_buffer->getMipWidth(level), Renderer::instance()->Last_Frame.hiz_buffer->getMipHeight(level)};
		vkCmdPushConstants(cmd_buffer, state.pass.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(VkExtent2D), &extent);

		uint32_t group_count_x = Renderer::instance()->getRenderTargetExtent().width / 32 + 1;
		uint32_t group_count_y = Renderer::instance()->getRenderTargetExtent().height / 32 + 1;
 		vkCmdDispatch(cmd_buffer, group_count_x, group_count_y, 1);

		{
			VkImageMemoryBarrier write_to_read{};
			write_to_read.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
			write_to_read.oldLayout           = VK_IMAGE_LAYOUT_GENERAL;
			write_to_read.newLayout           = VK_IMAGE_LAYOUT_GENERAL;
			write_to_read.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
			write_to_read.dstAccessMask       = VK_ACCESS_SHADER_READ_BIT;
			write_to_read.image               = *Renderer::instance()->Last_Frame.hiz_buffer;
			write_to_read.subresourceRange    = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, level, 1, 0, 1};
			write_to_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			write_to_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			vkCmdPipelineBarrier(
			    cmd_buffer,
			    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
			    VK_DEPENDENCY_BY_REGION_BIT,
			    0, nullptr,
			    0, nullptr,
			    1, &write_to_read);
		}
	}
}
}        // namespace Ilum::pass