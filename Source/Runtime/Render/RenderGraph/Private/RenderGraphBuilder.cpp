#include "RenderGraph/RenderGraphBuilder.hpp"

namespace Ilum
{
RenderGraphBuilder::RenderGraphBuilder(RHIContext *rhi_context) :
    p_rhi_context(rhi_context)
{
}

RenderGraphBuilder &RenderGraphBuilder::AddPass(RenderGraph &render_graph, const std::string &name, const std::string &category, BindPoint bind_point, const Variant &config, RenderGraph::RenderTask &&task, RenderGraph::BarrierTask &&barrier)
{
	render_graph.AddPass(name, category, bind_point, config, std::move(task), std::move(barrier));
	return *this;
}

bool RenderGraphBuilder::Validate(RenderGraphDesc &desc)
{
	bool result = true;

	std::set<size_t> used_pass;
	for (auto &[handle, pass] : desc.GetPasses())
	{
		if (desc.HasLink(handle))
		{
			used_pass.insert(handle);
			used_pass.insert(desc.GetPass(desc.LinkFrom(handle)).GetHandle());
		}
	}

	std::unordered_set<size_t> culling_passes;
	for (auto &[handle, pass] : desc.GetPasses())
	{
		if (used_pass.find(handle) == used_pass.end())
		{
			LOG_WARN("Pass <{}> is not used, culling it", pass.GetName());
			culling_passes.insert(handle);
		}
	}

	for (auto &pass : culling_passes)
	{
		desc.ErasePass(pass);
	}

	std::unordered_set<size_t> invalid_resources;
	for (auto &[pass_handle, pass] : desc.GetPasses())
	{
		for (auto &[pin_handle, pin] : pass.GetPins())
		{
			if (pin.attribute == RenderPassPin::Attribute::Input &&
			    !desc.HasLink(pin_handle))
			{
				invalid_resources.insert(pin_handle);
			}
		}
	}

	for (auto &resource : invalid_resources)
	{
		LOG_WARN("Input {} is not linked to a source", resource);
		result = false;
	}

	return result;
}

std::unique_ptr<RenderGraph> RenderGraphBuilder::Compile(RenderGraphDesc &desc, Renderer *renderer)
{
	// if (!Validate(desc))
	//{
	//	LOG_ERROR("Render graph compilation failed!");
	//	return nullptr;
	// }

	const std::unordered_map<BindPoint, RHIQueueFamily> queue_family_map = {
	    {BindPoint::None, RHIQueueFamily::Graphics},
	    {BindPoint::Rasterization, RHIQueueFamily::Graphics},
	    {BindPoint::RayTracing, RHIQueueFamily::Compute},
	    {BindPoint::Compute, RHIQueueFamily::Compute},
	    {BindPoint::CUDA, RHIQueueFamily::Compute},
	};

	struct ResourceState
	{
		RHIResourceState rhi_state;
		RHIQueueFamily   family;

		bool operator==(const ResourceState &lhs)
		{
			return this->rhi_state == lhs.rhi_state && this->family == lhs.family;
		}
	};

	// Sorting passes
	std::vector<size_t> ordered_passes;
	ordered_passes.reserve(desc.GetPasses().size());

	std::vector<size_t> pass_handles;

	std::map<size_t, std::map<uint32_t, ResourceState>> resource_lifetime;

	std::map<size_t, ResourceState> last_resource_state;

	for (auto &[pass_handle, pass] : desc.GetPasses())
	{
		pass_handles.push_back(pass_handle);
		for (auto &[pin_handle, pin] : pass.GetPins())
		{
			if (pin.attribute == RenderPassPin::Attribute::Output)
			{
				last_resource_state[pin_handle] = ResourceState{RHIResourceState::Undefined, queue_family_map.at(pass.GetBindPoint())};
			}
		}
	}

	std::vector<std::map<size_t, std::pair<ResourceState, ResourceState>>> resource_states;

	struct PassSemaphore
	{
		std::vector<RHISemaphore *> wait_semaphores;
		std::vector<RHISemaphore *> signal_semaphores;
	};

	std::map<size_t, PassSemaphore> pass_semaphores;        // [signal - wait]

	RenderGraphDesc tmp_desc = desc;

	std::set<size_t> collected_passes;

	// Create new render graph
	std::unique_ptr<RenderGraph> render_graph = std::make_unique<RenderGraph>(p_rhi_context);

	while (!pass_handles.empty())
	{
		uint32_t pass_idx = static_cast<uint32_t>(ordered_passes.size());

		for (auto iter = pass_handles.begin(); iter != pass_handles.end();)
		{
			auto &pass = desc.GetPass(*iter);

			if (!desc.HasLink(pass.GetHandle()) ||
			    collected_passes.find(desc.LinkFrom(pass.GetHandle())) != collected_passes.end())
			{
				std::map<size_t, std::pair<ResourceState, ResourceState>> pass_resource_state;

				for (auto &[pin_handle, pin] : pass.GetPins())
				{
					size_t resource_pin = (pin.attribute == RenderPassPin::Attribute::Output) ? pin_handle : (desc.HasLink(pin_handle) ? desc.LinkFrom(pin_handle) : ~0ull);
					if (resource_pin == ~0ull)
					{
						continue;
					}

					ResourceState resource_state = ResourceState{pin.resource_state, queue_family_map.at(pass.GetBindPoint())};

					// Record resource state
					if (pass.GetBindPoint() != BindPoint::CUDA)
					{
						pass_resource_state[resource_pin]         = std::make_pair(last_resource_state[resource_pin], resource_state);
						last_resource_state[resource_pin]         = resource_state;
						resource_lifetime[resource_pin][pass_idx] = resource_state;
					}
					else
					{
						ResourceState state = last_resource_state.find(resource_pin) != last_resource_state.end() ? last_resource_state[resource_pin] : ResourceState{RHIResourceState::Undefined, RHIQueueFamily::Graphics};

						resource_lifetime[resource_pin][pass_idx] = state;
						pass_resource_state[resource_pin]         = std::make_pair(state, state);
					}

					// Set resource info
					auto &resource = desc.GetPass(resource_pin).GetPin(resource_pin);
					if (resource.type == RenderPassPin::Type::Texture)
					{
						resource.texture.usage |= ResourceStateToTextureUsage(resource_state.rhi_state) | RHITextureUsage::Transfer;
					}
					else
					{
						resource.buffer.usage |= ResourceStateToBufferUsage(resource_state.rhi_state) | RHIBufferUsage::Transfer;
					}
				}

				// Add pass
				resource_states.push_back(pass_resource_state);
				ordered_passes.push_back(pass.GetHandle());
				collected_passes.insert(pass.GetHandle());
				pass_handles.erase(iter);
				break;
			}
			else
			{
				iter++;
			}
		}
	}

	std::set<size_t> alias_textures;

	// Register resource
	{
		// Collect texture alias info
		struct TexturePool
		{
			std::vector<size_t> handles;

			uint32_t start;
			uint32_t end;
		};

		std::vector<TexturePool> texture_pools;

		for (auto &[handle, resource_states] : resource_lifetime)
		{
			const auto &resource = desc.GetPass(handle).GetPin(handle);
			if (resource.type == RenderPassPin::Type::Texture)
			{
				bool alias = false;
				for (auto &pool : texture_pools)
				{
					if (pool.start > (--resource_states.end())->first ||
					    pool.end < resource_states.begin()->first)
					{
						pool.handles.push_back(handle);
						pool.start = std::min(pool.start, (--resource_states.end())->first);
						pool.end   = std::max(pool.end, resource_states.begin()->first);
						alias      = true;
						break;
					}
				}
				if (!alias)
				{
					texture_pools.push_back(TexturePool{{handle}, resource_states.begin()->first, (--resource_states.end())->first});
				}
			}
			else
			{
				std::set<size_t> handles = desc.LinkTo(handle);
				handles.insert(handle);
				render_graph->RegisterBuffer(RenderGraph::BufferCreateInfo{resource.buffer, handles});
			}
		}

		for (auto &pool : texture_pools)
		{
			std::vector<RenderGraph::TextureCreateInfo> texture_create_infos;
			texture_create_infos.reserve(pool.handles.size());
			for (auto &handle : pool.handles)
			{
				const auto      &resource = desc.GetPass(handle).GetPin(handle);
				std::set<size_t> handles  = desc.LinkTo(handle);
				handles.insert(handle);
				texture_create_infos.push_back(RenderGraph::TextureCreateInfo{resource.texture, handles});
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
			if (src.rhi_state == RHIResourceState::Undefined)
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

		for (auto &[handle, resource_states] : resource_lifetime)
		{
			const auto &resource = desc.GetPass(handle).GetPin(handle);
			if ((--resource_states.end())->second.rhi_state != RHIResourceState::Undefined)
			{
				if (resource.type == RenderPassPin::Type::Texture)
				{
					auto *texture = render_graph->GetTexture(handle);

					const auto &texture_desc = texture->GetDesc();

					texture_state_transitions.push_back(TextureStateTransition{
					    texture,
					    RHIResourceState::Undefined,
					    (--resource_states.end())->second.rhi_state,
					    TextureRange{
					        GetTextureDimension(texture_desc.width, texture_desc.height, texture_desc.depth, texture_desc.layers),
					        0,
					        texture_desc.mips,
					        0,
					        texture_desc.layers},
					    (--resource_states.end())->second.family,
					    (--resource_states.end())->second.family});
				}
				else
				{
					auto *buffer = render_graph->GetBuffer(handle);

					const auto &buffer_desc = buffer->GetDesc();

					buffer_state_transitions.push_back(BufferStateTransition{
					    buffer,
					    RHIResourceState::Undefined,
					    (--resource_states.end())->second.rhi_state});
				}
			}
		}

		render_graph->AddInitializeBarrier([=](RenderGraph &render_graph, RHICommand *graphics_cmd_buffer, RHICommand *compute_cmd_buffer) {
			std::vector<TextureStateTransition> graphics_texture_transitions;
			std::vector<TextureStateTransition> compute_texture_transitions;
			for (auto &transition : texture_state_transitions)
			{
				if (transition.dst_family == RHIQueueFamily::Compute)
				{
					compute_texture_transitions.push_back(transition);
				}
				else
				{
					graphics_texture_transitions.push_back(transition);
				}
			}
			graphics_cmd_buffer->ResourceStateTransition(graphics_texture_transitions, buffer_state_transitions);
			compute_cmd_buffer->ResourceStateTransition(compute_texture_transitions, {});
		});
	}

	struct ResourceStateTransitionInfo
	{
		size_t           handle;
		RHIResourceState src_state;
		RHIResourceState dst_state;
	};

	for (uint32_t i = 0; i < ordered_passes.size(); i++)
	{
		const auto &pass = desc.GetPass(ordered_passes[i]);

		std::vector<ResourceStateTransitionInfo> buffer_state_transition_infos;
		std::vector<ResourceStateTransitionInfo> texture_state_transition_infos;

		for (auto &[handle, resource_transition] : resource_states[i])
		{
			const auto &resource = desc.GetPass(handle).GetPin(handle);

			if (resource_transition.first == resource_transition.second)
			{
				continue;
			}

			if (resource.type == RenderPassPin::Type::Texture)
			{
				texture_state_transition_infos.push_back(ResourceStateTransitionInfo{
				    handle,
				    resource_transition.first.rhi_state,
				    resource_transition.second.rhi_state});

				/*texture_state_transitions.push_back(TextureStateTransition{
				    texture,
				    resource_transition.first.rhi_state,
				    resource_transition.second.rhi_state,
				    TextureRange{
				        GetTextureDimension(texture_desc.width, texture_desc.height, texture_desc.depth, texture_desc.layers),
				        0,
				        texture_desc.mips,
				        0,
				        texture_desc.layers},
				    i == 0 ? resource_transition.second.family : resource_transition.first.family,
				    resource_transition.second.family});*/
			}
			else
			{
				buffer_state_transition_infos.push_back(ResourceStateTransitionInfo{
				    handle,
				    resource_transition.first.rhi_state,
				    resource_transition.second.rhi_state});

				// buffer_state_transitions.push_back(BufferStateTransition{
				//     buffer,
				//     resource_transition.first.rhi_state,
				//     resource_transition.second.rhi_state});
			}
		}

		RenderGraph::RenderTask render_task;

		PluginManager::GetInstance().Call(fmt::format("shared/RenderPass/RenderPass.{}.{}.dll", pass.GetCategory(), pass.GetName()), "CreateCallback", &render_task, pass, *this, renderer);

		render_graph->AddPass(
		    pass.GetName(),
		    pass.GetCategory(),
		    pass.GetBindPoint(),
		    pass.GetConfig(),
		    std::move(render_task),
		    [=](RenderGraph &render_graph, RHICommand *cmd_buffer) {
			    std::vector<BufferStateTransition>  buffer_state_transitions(buffer_state_transition_infos.size());
			    std::vector<TextureStateTransition> texture_state_transitions(texture_state_transition_infos.size());

			    std::transform(
			        texture_state_transition_infos.begin(),
			        texture_state_transition_infos.end(),
			        texture_state_transitions.begin(),
			        [&](const ResourceStateTransitionInfo &info) {
				        auto       *texture = render_graph.GetTexture(info.handle);
				        const auto &desc    = texture->GetDesc();
				        return TextureStateTransition{
				            texture,
				            info.src_state,
				            info.dst_state,
				            TextureRange{
				                GetTextureDimension(desc.width, desc.height, desc.depth, desc.layers),
				                0,
				                desc.mips,
				                0,
				                desc.layers}};
			        });

			    std::transform(
			        buffer_state_transition_infos.begin(),
			        buffer_state_transition_infos.end(),
			        buffer_state_transitions.begin(),
			        [&](const ResourceStateTransitionInfo &info) {
				        auto       *buffer = render_graph.GetBuffer(info.handle);
				        const auto &desc   = buffer->GetDesc();
				        return BufferStateTransition{
				            buffer,
				            info.src_state,
				            info.dst_state};
			        });

			    cmd_buffer->ResourceStateTransition(texture_state_transitions, buffer_state_transitions);
		    });
	}

	return render_graph;
}
}        // namespace Ilum