#pragma once

#include <Core/Core.hpp>
#include <RHI/RHISampler.hpp>

#include "MaterialGraph.hpp"
#include "MaterialNode.hpp"

namespace Ilum
{
struct MaterialCompilationContext
{
	struct BSDF
	{
		std::string name = "";
		std::string type = "";

		std::string initialization = "";

		template <typename Archive>
		void serialize(Archive &archive)
		{
			archive(name, type, initialization);
		}
	};

	std::vector<std::string> variables;

	std::map<std::string, std::string> textures;

	std::map<std::string, SamplerDesc> samplers;

	std::vector<BSDF> bsdfs;

	struct
	{
		std::string bsdf = "";
	} output;

	std::unordered_set<size_t> finish_nodes;

	template <typename Archive>
	void serialize(Archive &archive)
	{
		archive(variables, textures, samplers, bsdfs, output.bsdf, finish_nodes);
	}

	void Reset()
	{
		variables.clear();
		textures.clear();
		samplers.clear();
		bsdfs.clear();
		finish_nodes.clear();
	}

	bool IsCompiled(const MaterialNodeDesc &desc)
	{
		if (finish_nodes.find(desc.GetHandle()) == finish_nodes.end())
		{
			finish_nodes.insert(desc.GetHandle());
			return false;
		}
		return true;
	}

	template <typename T>
	inline void SetParameter(const std::string &name, std::map<std::string, std::string> &parameters, T var)
	{
	}

	template <>
	inline void SetParameter(const std::string &name, std::map<std::string, std::string> &parameters, float var)
	{
		parameters[name] = fmt::format("{}", var);
	}

	template <>
	inline void SetParameter(const std::string &name, std::map<std::string, std::string> &parameters, bool var)
	{
		parameters[name] = fmt::format("{}", var);
	}

	template <>
	inline void SetParameter(const std::string &name, std::map<std::string, std::string> &parameters, glm::vec3 var)
	{
		parameters[name] = fmt::format("float3({}, {}, {})", var.x, var.y, var.z);
	}

	template <typename T>
	inline bool HasParameter(std::map<std::string, std::string> &parameters, const MaterialNodePin &node_pin, const MaterialGraphDesc &graph_desc, ResourceManager *manager, MaterialCompilationContext *context)
	{
		if (graph_desc.HasLink(node_pin.handle))
		{
			auto &src_node = graph_desc.GetNode(graph_desc.LinkFrom(node_pin.handle));
			src_node.EmitHLSL(graph_desc, manager, context);
			auto &src_pin = src_node.GetPin(graph_desc.LinkFrom(node_pin.handle));
			if (src_pin.type != node_pin.type && src_pin.type == MaterialNodePin::Type::Float)
			{
				parameters[node_pin.name] = fmt::format("float3(S_{}, S_{}, S_{})", src_pin.handle, src_pin.handle, src_pin.handle);
			}
			else if (src_pin.type != node_pin.type && node_pin.type == MaterialNodePin::Type::Float)
			{
				parameters[node_pin.name] = fmt::format("S_{}.x", src_pin.handle);
			}
			else
			{
				parameters[node_pin.name] = fmt::format("S_{}", src_pin.handle);
			}
			return true;
		}
		return false;
	}

	template <typename T>
	inline void SetParameter(std::map<std::string, std::string> &parameters, const MaterialNodePin &node_pin, const MaterialGraphDesc &graph_desc, ResourceManager *manager, MaterialCompilationContext *context)
	{
		if (!HasParameter<T>(parameters, node_pin, graph_desc, manager, context))
		{
			SetParameter<T>(node_pin.name, parameters, *node_pin.variant.Convert<T>());
		}
	}
};
}        // namespace Ilum