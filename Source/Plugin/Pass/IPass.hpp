#pragma once

#include <Core/Core.hpp>
#include <RenderGraph/RenderGraph.hpp>
#include <Renderer/Renderer.hpp>

namespace Ilum
{
struct RenderPassDesc;
class RenderGraphBuilder;
class Renderer;
}        // namespace Ilum

struct ImGuiContext;

using namespace Ilum;

template <typename T>
class IPass
{
  public:
	IPass() = default;

	~IPass() = default;

	static T &GetInstance()
	{
		static T instance;
		return instance;
	}

	virtual void CreateDesc(RenderPassDesc *desc)
	{
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
		EXPORT_API void CreateDesc(RenderPassDesc *desc)                                                                                           \
		{                                                                                                                                          \
			Pass::GetInstance().CreateDesc(desc);                                                                                                  \
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