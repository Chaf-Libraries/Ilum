#pragma once

#include <Core/Core.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <RenderGraph/RenderGraphBlackboard.hpp>
#include <RenderGraph/RenderGraphBuilder.hpp>
#include <Renderer/RenderData.hpp>
#include <Renderer/Renderer.hpp>

#include <imgui.h>

namespace Ilum
{
class RenderPassDesc;
class RenderGraphBuilder;
class Renderer;
}        // namespace Ilum

struct ImGuiContext;

using namespace Ilum;

template <typename T>
class RenderPass
{
  public:
	RenderPass() = default;

	~RenderPass() = default;

	static T &GetInstance()
	{
		static T instance;
		return instance;
	}

	virtual RenderPassDesc Create(size_t &handle)
	{
		return RenderPassDesc{};
	}

	virtual void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer)
	{
	}

	virtual void OnImGui(Variant *config)
	{
	}
};

#define CONFIGURATION_PASS(Pass)                                                                                                                   \
	extern "C"                                                                                                                                     \
	{                                                                                                                                              \
		EXPORT_API void Create(RenderPassDesc *desc, size_t &handle)                                                                           \
		{                                                                                                                                          \
			*desc = Pass::GetInstance().Create(handle);                                                                                        \
		}                                                                                                                                          \
		EXPORT_API void CreateCallback(RenderGraph::RenderTask *task, const RenderPassDesc &desc, RenderGraphBuilder &builder, Renderer *renderer) \
		{                                                                                                                                          \
			Pass::GetInstance().CreateCallback(task, desc, builder, renderer);                                                                     \
		}                                                                                                                                          \
		EXPORT_API void OnImGui(Variant *config, ImGuiContext *context)                                                                            \
		{                                                                                                                                          \
			ImGui::SetCurrentContext(context);                                                                                                     \
			Pass::GetInstance().OnImGui(config);                                                                                                   \
		}                                                                                                                                          \
	}