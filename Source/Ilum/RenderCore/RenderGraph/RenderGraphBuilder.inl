#pragma once

#include "RenderGraphBuilder.hpp"

namespace Ilum
{
template <typename... Args>
std::unique_ptr<RenderGraph> RenderGraphBuilder::Compile(RenderGraphDesc &desc, Args &&...args)
{
	if (!Validate(desc))
	{
		LOG_ERROR("Render graph compilation failed!");
		return nullptr;
	}

	// Sorting passes
	std::vector<RenderPassDesc> ordered_passes;
	ordered_passes.reserve(desc.passes.size());

	std::map<RGHandle, std::map<uint32_t, RHIResourceState>> resource_lifetime;
	std::map<RGHandle, RHIResourceState>                     last_resource_state;
	for (auto &[texture_handle, texture_desc] : desc.textures)
	{
		last_resource_state[texture_handle] = RHIResourceState::Undefined;
	}
	for (auto &[buffer_handle, buffer_desc] : desc.buffers)
	{
		last_resource_state[buffer_handle] = RHIResourceState::Undefined;
	}

	std::vector<std::map<RGHandle, std::pair<RHIResourceState, RHIResourceState>>> resource_states;

	RenderGraphDesc tmp_desc = desc;

	std::set<RGHandle> collected_passes;

	while (!tmp_desc.passes.empty())
	{
		uint32_t pass_idx = static_cast<uint32_t>(ordered_passes.size());

		for (auto iter = tmp_desc.passes.begin(); iter != tmp_desc.passes.end();)
		{
			if (!iter->second.prev_pass.IsValid() || collected_passes.find(iter->second.prev_pass) != collected_passes.end())
			{
				std::map<RGHandle, std::pair<RHIResourceState, RHIResourceState>> pass_resource_state;
				for (auto &[name, resource] : iter->second.resources)
				{
					if (resource.handle.IsValid())
					{
						resource_lifetime[resource.handle][pass_idx] = resource.state;
						pass_resource_state[resource.handle]         = std::make_pair(last_resource_state[resource.handle], resource.state);
						last_resource_state[resource.handle]         = resource.state;

						if (desc.textures.find(resource.handle) != desc.textures.end())
						{
							desc.textures[resource.handle].usage |= ResourceStateToTextureUsage(resource.state);
							if (iter->second.bind_point == BindPoint::CUDA)
							{
								desc.textures[resource.handle].external = true;
							}
						}
						else if (desc.buffers.find(resource.handle) != desc.buffers.end())
						{
							desc.buffers[resource.handle].usage |= ResourceStateToBufferUsage(resource.state);
						}
					}
				}
				resource_states.push_back(pass_resource_state);
				ordered_passes.push_back(iter->second);
				collected_passes.insert(iter->first);
				iter = tmp_desc.passes.erase(iter);
				break;
			}
			else
			{
				iter++;
			}
		}
	}

	std::set<RGHandle> alias_textures;

	// Create new render graph
	std::unique_ptr<RenderGraph> render_graph = std::make_unique<RenderGraph>(p_rhi_context);

	// Register resource
	{
		// Collect texture alias info
		struct TexturePool
		{
			std::vector<RGHandle> handles;

			uint32_t start;
			uint32_t end;
		};

		std::vector<TexturePool> texture_pools;

		for (auto &[handle, resource] : resource_lifetime)
		{
			if (desc.textures.find(handle) != desc.textures.end())
			{
				bool alias = false;
				for (auto &pool : texture_pools)
				{
					if (desc.textures[handle].external)
					{
						render_graph->RegisterTexture(RenderGraph::TextureCreateInfo{desc.textures[handle], handle});
					}
					else
					{
						if (pool.start > (--resource.end())->first ||
						    pool.end < resource.begin()->first)
						{
							pool.handles.push_back(handle);
							pool.start = std::min(pool.start, (--resource.end())->first);
							pool.end   = std::max(pool.end, resource.begin()->first);
							alias      = true;
							break;
						}
					}
				}
				if (!alias)
				{
					texture_pools.push_back(TexturePool{{handle}, resource.begin()->first, (--resource.end())->first});
				}
			}
			else
			{
				render_graph->RegisterBuffer(RenderGraph::BufferCreateInfo{desc.buffers[handle], handle});
			}
		}

		for (auto &pool : texture_pools)
		{
			std::vector<RenderGraph::TextureCreateInfo> texture_create_infos;
			texture_create_infos.reserve(pool.handles.size());
			for (auto &handle : pool.handles)
			{
				texture_create_infos.push_back(RenderGraph::TextureCreateInfo{desc.textures[handle], handle});
				if (pool.handles.size() > 1)
				{
					alias_textures.insert(handle);
				}
			}
			render_graph->RegisterTexture(texture_create_infos);
		}
	}

	// Resource state tracking
	for (auto &resource_state : resource_states)
	{
		for (auto &[resource_handle, state_transition] : resource_state)
		{
			auto &[src, dst] = state_transition;
			if (src == RHIResourceState::Undefined)
			{
				if (alias_textures.find(resource_handle) == alias_textures.end())
				{
					src = (--resource_lifetime[resource_handle].end())->second;
				}
			}
		}
	}

	// Initialize Barrier
	{
		std::vector<BufferStateTransition>  buffer_state_transitions;
		std::vector<TextureStateTransition> texture_state_transitions;

		for (auto &[handle, resource] : resource_lifetime)
		{
			if ((--resource.end())->second != RHIResourceState::Undefined)
			{
				if (desc.textures.find(handle) != desc.textures.end())
				{
					auto *texture = render_graph->GetTexture(handle);

					const auto &texture_desc = texture->GetDesc();

					texture_state_transitions.push_back(TextureStateTransition{
					    texture,
					    RHIResourceState::Undefined,
					    (--resource.end())->second,
					    TextureRange{
					        GetTextureDimension(texture_desc.width, texture_desc.height, texture_desc.depth, texture_desc.layers),
					        0,
					        texture_desc.mips,
					        0,
					        texture_desc.layers}});
				}
				else
				{
					auto *buffer = render_graph->GetBuffer(handle);

					const auto &buffer_desc = buffer->GetDesc();

					buffer_state_transitions.push_back(BufferStateTransition{
					    buffer,
					    RHIResourceState::Undefined,
					    (--resource.end())->second});
				}
			}
		}

		render_graph->AddInitializeBarrier([=](RenderGraph &render_graph, RHICommand *cmd_buffer) {
			cmd_buffer->ResourceStateTransition(texture_state_transitions, buffer_state_transitions);
		});
	}

	for (uint32_t i = 0; i < ordered_passes.size(); i++)
	{
		const auto &pass = ordered_passes[i];

		std::vector<BufferStateTransition>  buffer_state_transitions;
		std::vector<TextureStateTransition> texture_state_transitions;

		for (auto &[handle, resource_transition] : resource_states[i])
		{
			if (resource_transition.first == resource_transition.second)
			{
				continue;
			}

			if (desc.textures.find(handle) != desc.textures.end())
			{
				auto *texture = render_graph->GetTexture(handle);

				const auto &texture_desc = texture->GetDesc();

				texture_state_transitions.push_back(TextureStateTransition{
				    texture,
				    resource_transition.first,
				    resource_transition.second,
				    TextureRange{
				        GetTextureDimension(texture_desc.width, texture_desc.height, texture_desc.depth, texture_desc.layers),
				        0,
				        texture_desc.mips,
				        0,
				        texture_desc.layers}});
			}
			else
			{
				auto *buffer = render_graph->GetBuffer(handle);

				const auto &buffer_desc = buffer->GetDesc();

				buffer_state_transitions.push_back(BufferStateTransition{
				    buffer,
				    resource_transition.first,
				    resource_transition.second});
			}
		}

		auto pass_type   = rttr::type::get_by_name("Ilum::" + pass.name);
		auto render_task = pass_type.get_method("Create").invoke(pass_type.create(), pass, *this, std::forward<Args>(args)...);

		AddPass(
		    *render_graph,
		    pass.name,
			pass.bind_point,
			pass.config,
		    std::move(render_task.convert<RenderGraph::RenderTask>()),
		    [=](RenderGraph &render_graph, RHICommand *cmd_buffer) {
			    cmd_buffer->ResourceStateTransition(texture_state_transitions, buffer_state_transitions);
		    });
	}

	return render_graph;
}
}        // namespace Ilum