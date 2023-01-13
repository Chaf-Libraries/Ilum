#include "RenderGraph/RenderGraphBuilder.hpp"

namespace Ilum
{
RenderGraphBuilder::RenderGraphBuilder(RHIContext *rhi_context) :
    p_rhi_context(rhi_context)
{
}

RenderGraphBuilder &RenderGraphBuilder::AddPass(RenderGraph &render_graph, const std::string &name, BindPoint bind_point, const Variant &config, RenderGraph::RenderTask &&task, RenderGraph::BarrierTask &&barrier)
{
	render_graph.AddPass(name, bind_point, config, std::move(task), std::move(barrier));
	return *this;
}

bool RenderGraphBuilder::Validate(RenderGraphDesc &desc)
{
	// Culling unused pass
	{
		std::set<RGHandle> used_pass;
		for (auto &[handle, pass] : desc.passes)
		{
			if (pass.prev_pass.IsValid())
			{
				used_pass.insert(handle);
				used_pass.insert(pass.prev_pass);
			}

			for (auto &[name, resource] : pass.resources)
			{
				if (!resource.handle.IsValid() &&
				    resource.attribute == RenderResourceDesc::Attribute::Write)
				{
					LOG_ERROR("Pass <{}>'s output <{}> should be bound with a resource", pass.name, name);
					return false;
				}
			}
		}

		for (auto iter = desc.passes.begin(); iter != desc.passes.end();)
		{
			if (used_pass.find(iter->first) == used_pass.end())
			{
				LOG_WARN("Pass <{}> is not used, culling it", iter->second.name);
				iter = desc.passes.erase(iter);
			}
			else
			{
				iter++;
			}
		}
	}

	// Culling unused texture/buffer and check for loop
	{
		std::set<RGHandle> used_resource;
		size_t             prev_pass_count = 0;
		for (auto &[handle, pass] : desc.passes)
		{
			for (auto &[name, resource] : pass.resources)
			{
				used_resource.insert(resource.handle);
			}

			if (pass.prev_pass.IsValid())
			{
				prev_pass_count++;
			}
		}

		if (prev_pass_count == desc.passes.size())
		{
			LOG_ERROR("Render graph contains a loop!");
			return false;
		}

		for (auto iter = desc.textures.begin(); iter != desc.textures.end();)
		{
			if (used_resource.find(iter->first) == used_resource.end())
			{
				LOG_WARN("Texture <{}> is not used, culling it", iter->second.name);
				iter = desc.textures.erase(iter);
			}
			else
			{
				iter++;
			}
		}

		for (auto iter = desc.buffers.begin(); iter != desc.buffers.end();)
		{
			if (used_resource.find(iter->first) == used_resource.end())
			{
				LOG_WARN("Buffer <{}> is not used, culling it", iter->second.name);
				iter = desc.buffers.erase(iter);
			}
			else
			{
				iter++;
			}
		}
	}

	return true;
}

std::unique_ptr<RenderGraph> RenderGraphBuilder::Compile()
{
	return nullptr;
}
}        // namespace Ilum