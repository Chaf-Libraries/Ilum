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
	RenderGraphBuilder() = default;

	~RenderGraphBuilder() = default;

	RenderGraphBuilder &AddPass(const std::string &name, std::function<void(RenderGraph &)> &&task)
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

		{
			RenderGraphDesc tmp_desc = desc;

			while (!tmp_desc.passes.empty())
			{
				// Collect pass without input
				for (auto iter = tmp_desc.passes.begin(); iter != tmp_desc.passes.end();)
				{
					auto &[handle, pass] = *iter;
					bool found           = true;
					for (auto &[name, read] : pass.reads)
					{
						for (auto &edge : tmp_desc.edges)
						{
							auto [src, dst] = RenderGraphDesc::DecodeEdge(edge.first, edge.second);
							if (read.second == dst)
							{
								found = false;
								break;
							}
						}
					}
					if (found)
					{
						ordered_passes.push_back(pass);
						iter = tmp_desc.passes.erase(iter);
						// Remove all its output
						std::set<RGHandle> culled_nodes;
						for (auto &[name, write] : ordered_passes.back().writes)
						{
							for (auto edge_iter = tmp_desc.edges.begin(); edge_iter != tmp_desc.edges.end();)
							{
								auto [src, dst] = RenderGraphDesc::DecodeEdge(edge_iter->first, edge_iter->second);
								if (write.second == src)
								{
									culled_nodes.insert(dst);
									edge_iter = tmp_desc.edges.erase(edge_iter);
								}
								else
								{
									edge_iter++;
								}
							}
						}
						for (auto edge_iter = tmp_desc.edges.begin(); edge_iter != tmp_desc.edges.end();)
						{
							auto [src, dst] = RenderGraphDesc::DecodeEdge(edge_iter->first, edge_iter->second);
							if (culled_nodes.find(src) != culled_nodes.end())
							{
								edge_iter = tmp_desc.edges.erase(edge_iter);
							}
							else
							{
								edge_iter++;
							}
						}
						continue;
					}
					iter++;
				}
			}
		}

		// 
		{

		}

		// for (auto &[handle, pass] : desc.passes)
		//{
		//	std::vector<rttr::argument> arguments;
		//	variadic_vector_emplace(arguments, pass, *this, std::forward<Args>(args)...);
		//	rttr::type::invoke(fmt::format("{}_Creation", pass.name).c_str(), arguments);
		// }

		return Compile();
	}
};
}        // namespace Ilum