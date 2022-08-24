#include "RenderGraphBuilder.hpp"

namespace Ilum
{
bool RenderGraphBuilder::Validate(RenderGraphDesc &desc)
{
	std::set<RGHandle> cull_nodes;
	std::set<RGHandle> src_nodes;
	std::set<RGHandle> dst_nodes;

	for (auto &[src, dst] : desc.edges)
	{
		auto [src_handle, dst_handle] = RenderGraphDesc::DecodeEdge(src, dst);
		src_nodes.insert(src_handle);
		dst_nodes.insert(dst_handle);
	}

	// Texture must be written
	for (auto iter = desc.textures.begin(); iter != desc.textures.end();)
	{
		auto &[handle, texture] = *iter;
		if (dst_nodes.find(handle) == dst_nodes.end())
		{
			if (src_nodes.find(handle) == src_nodes.end())
			{
				LOG_INFO("Texture ({}) is never been used, culling it.", texture.name);
				cull_nodes.insert(handle);
				iter = desc.textures.erase(iter);
				continue;
			}
			else
			{
				LOG_ERROR("Texture ({}) has been read, but never been written!", texture.name);
				return false;
			}
		}
		iter++;
	}

	// Buffer must be written
	for (auto iter = desc.buffers.begin(); iter != desc.buffers.end();)
	{
		auto &[handle, buffer] = *iter;
		if (dst_nodes.find(handle) == dst_nodes.end())
		{
			if (src_nodes.find(handle) == src_nodes.end())
			{
				LOG_INFO("Buffer ({}) is never been used, culling it.", buffer.name);
				cull_nodes.insert(handle);
				iter = desc.buffers.erase(iter);
				continue;
			}
			else
			{
				LOG_ERROR("Buffer ({}) has been read, but never been written!", buffer.name);
				return false;
			}
		}
		iter++;
	}

	// Pass can have null texture to read by can't have null texture to write
	for (auto iter = desc.passes.begin(); iter != desc.passes.end();)
	{
		auto &[handle, pass] = *iter;
		size_t count         = 0;
		for (auto &[name, write] : pass.writes)
		{
			if (src_nodes.find(write.handle) == src_nodes.end())
			{
				count++;
			}
		}
		if (count == pass.writes.size() && pass.writes.size() != 0)
		{
			LOG_INFO("Pass ({}) is never been used, culling it", pass.name);
			cull_nodes.insert(handle);
			for (auto &[name, write] : pass.writes)
			{
				cull_nodes.insert(write.handle);
			}
			for (auto &[name, read] : pass.reads)
			{
				cull_nodes.insert(read.handle);
			}
			iter = desc.passes.erase(iter);
			continue;
		}
		else if (count != 0)
		{
			LOG_ERROR("Pass {} writes to a null texture!", pass.name);
			return false;
		}
		iter++;
	}

	// Clear all culled nodes
	for (auto iter = desc.edges.begin(); iter != desc.edges.end();)
	{
		auto &[src, dst]              = *iter;
		auto [src_handle, dst_handle] = RenderGraphDesc::DecodeEdge(src, dst);
		if (cull_nodes.find(src_handle) != cull_nodes.end() ||
		    cull_nodes.find(dst_handle) != cull_nodes.end())
		{
			iter = desc.edges.erase(iter);
		}
		else
		{
			iter++;
		}
	}

	return true;
}
}        // namespace Ilum