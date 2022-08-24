#pragma once

#include "RenderGraph.hpp"

namespace Ilum
{
class RenderGraph;

template <typename T>
void variadic_vector_emplace(std::vector<T> &)
{}

template <typename T, typename First, typename... Args>
void variadic_vector_emplace(std::vector<T> &v, First &&first, Args &&...args)
{
	v.emplace_back(std::forward<First>(first));
	variadic_vector_emplace(v, std::forward<Args>(args)...);
}

class RenderGraphBuilder
{
  public:
	RenderGraphBuilder(RHIContext *rhi_context) :
	    p_rhi_context(rhi_context)
	{
	}

	~RenderGraphBuilder() = default;

	RenderGraphBuilder &AddPass(const std::string &name, std::function<void(RenderGraph &, RHICommand *)> &&task)
	{
		return *this;
	}

	bool Validate(RenderGraphDesc &desc);

	std::unique_ptr<RenderGraph> Compile()
	{
		return nullptr;
	}

	template <typename... Args>
	std::unique_ptr<RenderGraph> Compile(RenderGraphDesc &desc, Args &&...args)
	{
		if (!Validate(desc))
		{
			return nullptr;
		}

		// Sorting Passes
		std::vector<RenderPassDesc> ordered_passes;
		ordered_passes.reserve(desc.passes.size());

		RenderGraphDesc tmp_desc = desc;

		std::map<RGHandle, std::vector<RGHandle>> resource_references;
		std::vector<std::vector<std::pair<RGHandle, RHIResourceState>>> resource_states;
		std::map<RGHandle, std::pair<uint32_t, uint32_t>> resource_life;

		while (!tmp_desc.passes.empty())
		{
			for (auto iter = tmp_desc.passes.begin(); iter != tmp_desc.passes.end();)
			{
				auto &[handle, pass] = *iter;

				bool found = true;

				for (auto &[name, read] : pass.reads)
				{
					for (auto &edge : tmp_desc.edges)
					{
						auto [src, dst] = RenderGraphDesc::DecodeEdge(edge.first, edge.second);
						if (read.handle == dst)
						{
							found = false;
							break;
						}
					}
					if (!found)
					{
						break;
					}
				}

				// Not a root, continue to search next one
				if (!found)
				{
					iter++;
					continue;
				}

				for (auto edge_iter = tmp_desc.edges.begin(); edge_iter != tmp_desc.edges.end();)
				{
					auto [src_handle, dst_handle] = RenderGraphDesc::DecodeEdge(edge_iter->first, edge_iter->second);

					for (auto &[name, write] : pass.writes)
					{
						if (write == src_handle)
						{
							resource_references[dst_handle].push_back(write);
						}
					}

					for (auto &[name, read] : pass.reads)
					{
						if (read == dst_handle)
						{
							resource_references[src_handle].push_back(read);
						}
					}

					edge_iter++;
				}

				ordered_passes.push_back(pass);
				iter = tmp_desc.passes.erase(iter);
			}
		}

		//while (!tmp_desc.passes.empty())
		//{
		//	// Collect pass without input
		//	for (auto iter = tmp_desc.passes.begin(); iter != tmp_desc.passes.end();)
		//	{
		//		auto &[handle, pass] = *iter;
		//		bool found           = true;
		//		for (auto &[name, read] : pass.reads)
		//		{
		//			for (auto &edge : tmp_desc.edges)
		//			{
		//				auto [src, dst] = RenderGraphDesc::DecodeEdge(edge.first, edge.second);
		//				if (read.handle == dst)
		//				{
		//					found = false;
		//					break;
		//				}
		//			}
		//		}
		//		if (found)
		//		{
		//			ordered_passes.push_back(pass);
		//			iter = tmp_desc.passes.erase(iter);
		//			// Remove all its output
		//			std::set<RGHandle> culled_nodes;
		//			for (auto &[name, write] : ordered_passes.back().writes)
		//			{
		//				for (auto edge_iter = tmp_desc.edges.begin(); edge_iter != tmp_desc.edges.end();)
		//				{
		//					auto [src, dst] = RenderGraphDesc::DecodeEdge(edge_iter->first, edge_iter->second);
		//					if (write.handle == src)
		//					{
		//						culled_nodes.insert(dst);
		//						edge_iter = tmp_desc.edges.erase(edge_iter);
		//						resource_attributes[dst].push_back(src);
		//					}
		//					else
		//					{
		//						edge_iter++;
		//					}
		//				}
		//			}
		//			for (auto edge_iter = tmp_desc.edges.begin(); edge_iter != tmp_desc.edges.end();)
		//			{
		//				auto [src, dst] = RenderGraphDesc::DecodeEdge(edge_iter->first, edge_iter->second);
		//				if (culled_nodes.find(src) != culled_nodes.end())
		//				{
		//					edge_iter = tmp_desc.edges.erase(edge_iter);
		//					resource_attributes[dst].push_back(dst);
		//				}
		//				else
		//				{
		//					edge_iter++;
		//				}
		//			}
		//			continue;
		//		}
		//		iter++;
		//	}
		//}

		//// Handle resource lifetime
		// std::vector<std::map<RGHandle, RHIResourceState>> texture_states(ordered_passes.size());
		// std::vector<std::map<RGHandle, RHIResourceState>> buffer_states(ordered_passes.size());

		// std::map<RGHandle, std::pair<uint32_t, uint32_t>> texture_lifetime;
		// std::map<RGHandle, std::pair<uint32_t, uint32_t>> buffer_lifetime;

		// std::map<RGHandle, std::vector<RGHandle>> texture_edges;
		// std::map<RGHandle, std::vector<RGHandle>> buffer_edges;

		// for (uint32_t i = 0; i < ordered_passes.size(); i++)
		//{
		//	for (auto &[src, dst] : desc.edges)
		//	{
		//		auto [src_handle, dst_handle] = RenderGraphDesc::DecodeEdge(src, dst);

		//		for (auto &[name, write] : ordered_passes[i].writes)
		//		{
		//			if (src_handle == write.handle)
		//			{
		//				if (write.type == RenderPassDesc::ResourceInfo::Type::Texture)
		//				{
		//					texture_states[i][dst_handle] = write.state;
		//					texture_edges[dst_handle].push_back(src_handle);
		//					if (texture_lifetime.find(dst_handle) != texture_lifetime.end())
		//					{
		//						texture_lifetime[dst_handle].first = i;
		//					}
		//					else
		//					{
		//						texture_lifetime[dst_handle].second = i;
		//					}
		//				}
		//				else if (write.type == RenderPassDesc::ResourceInfo::Type::Buffer)
		//				{
		//					buffer_states[i][dst_handle] = write.state;
		//					buffer_edges[dst_handle].push_back(src_handle);
		//					if (buffer_lifetime.find(dst_handle) != buffer_lifetime.end())
		//					{
		//						buffer_lifetime[dst_handle].first = i;
		//					}
		//					else
		//					{
		//						buffer_lifetime[dst_handle].second = i;
		//					}
		//				}
		//			}
		//		}

		//		for (auto &[name, read] : ordered_passes[i].reads)
		//		{
		//			if (dst_handle == read.handle)
		//			{
		//				if (read.type == RenderPassDesc::ResourceInfo::Type::Texture)
		//				{
		//					texture_states[i][src_handle] = read.state;
		//					texture_edges[src_handle].push_back(dst_handle);
		//					if (texture_lifetime.find(src_handle) != texture_lifetime.end())
		//					{
		//						texture_lifetime[src_handle].first  = i;
		//						texture_lifetime[src_handle].second = i;
		//					}
		//					else
		//					{
		//						texture_lifetime[src_handle].second = i;
		//					}
		//				}
		//				else if (read.type == RenderPassDesc::ResourceInfo::Type::Buffer)
		//				{
		//					buffer_states[i][src_handle] = read.state;
		//					buffer_edges[src_handle].push_back(dst_handle);
		//					if (buffer_lifetime.find(src_handle) != buffer_lifetime.end())
		//					{
		//						buffer_lifetime[src_handle].first  = i;
		//						buffer_lifetime[src_handle].second = i;
		//					}
		//					else
		//					{
		//						buffer_lifetime[src_handle].second = i;
		//					}
		//				}
		//			}
		//		}
		//	}
		//}

		// std::unique_ptr<RenderGraph> render_graph = std::make_unique<RenderGraph>(p_rhi_context);

		//// Initialize barrier
		// render_graph->AddInitializeBarrier([=](RenderGraph &rg, RHICommand *cmd_buffer) {
		//	std::vector<TextureStateTransition> texture_state_transitions;
		//	std::vector<BufferStateTransition>  buffer_state_transitions;
		//	for (auto &[handle, state] : texture_states[0])
		//	{
		//		RHITexture *texture = rg.GetTexture(handle);
		//		texture_state_transitions.push_back(TextureStateTransition{
		//		    texture,
		//		    RHIResourceState::Undefined,
		//		    state,
		//		    TextureRange{texture->GetDesc().depth == 1 ? RHITextureDimension::Texture2D : RHITextureDimension::Texture3D, 0, texture->GetDesc().mips, 0, texture->GetDesc().layers}});
		//	}
		//	for (auto &[handle, state] : buffer_states[0])
		//	{
		//		buffer_state_transitions.push_back(BufferStateTransition{rg.GetBuffer(handle), RHIResourceState::Undefined, state});
		//	}
		//	cmd_buffer->ResourceStateTransition(texture_state_transitions, buffer_state_transitions);
		// });

		//// Create buffer resource
		//// TODO: memory alias
		// for (auto &buffer : buffer_lifetime)
		//{
		//	// render_graph->RegisterBuffer(RenderGraph::BufferCreateInfo{desc.buffers.at(buffer), buffer_edges.at(buffer)});
		// }

		// std::map<RGHandle, RHIResourceState> last_states;

		// for (uint32_t i = 0; i < ordered_passes.size(); i++)
		//{
		//	std::vector<rttr::argument> arguments;
		//	variadic_vector_emplace(arguments, ordered_passes[i], *this, std::forward<Args>(args)...);
		//	render_graph->AddPass(
		//	    ordered_passes[i].name,
		//	    rttr::type::invoke(fmt::format("{}_Creation", ordered_passes[i].name).c_str(), arguments).convert<std::function<void(RenderGraph &, RHICommand *)>>(),
		//	    [=](RenderGraph &rg, RHICommand *cmd_buffer) {

		//	    });
		//}

		// for (auto &[handle, pass] : desc.passes)
		//{
		//	std::vector<rttr::argument> arguments;
		//	variadic_vector_emplace(arguments, pass, *this, std::forward<Args>(args)...);
		//	rttr::type::invoke(fmt::format("{}_Creation", pass.name).c_str(), arguments);
		// }

		return Compile();
	}

  private:
	RHIContext *p_rhi_context = nullptr;
};
}        // namespace Ilum