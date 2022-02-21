#include <Ilum/Engine/Context.hpp>
#include <Ilum/Engine/Engine.hpp>
#include <Ilum/Scene/Scene.hpp>
#include <Ilum/Scene/System.hpp>

#include <Ilum/Editor/Editor.hpp>
#include <Ilum/Editor/Panel.hpp>

#include <Graphics/Device/Window.hpp>
#include <Graphics/RenderContext.hpp>
#include <Graphics/Resource/Image.hpp>

#include <Resource/ResourceCache.hpp>

#include <Render/RenderGraph/RenderGraph.hpp>
#include <Render/RenderGraph/ResourceNode.hpp>

#include <Core/JobSystem/JobSystem.hpp>

#include <imgui.h>
#include <imnodes/imnodes.h>

namespace Ilum::Render
{
class TestNode : public RenderNode
{
  public:
	virtual void OnImGui() override;

	virtual void OnImNode() override;

  private:
	struct Pin: public PinDesc
	{
		virtual size_t Hash() override
		{
			return 0;
		}

		virtual int32_t GetNode() override
		{
			return m_uuid;
		}

		int32_t node = m_uuid;
	}m_pin;
};
};        // namespace Ilum::Render

namespace Ilum::panel
{
class RGEditor : public Panel
{
  public:
	RGEditor()
	{
		m_name = "Render Graph Editor";
	}

	~RGEditor() = default;

	virtual void draw(float delta_time) override
	{
		ImGui::Begin(m_name.c_str(), &active);

		ImNodes::BeginNodeEditor();

		ImNodes::BeginNode(1);

		ImNodes::BeginNodeTitleBar();
		ImGui::TextUnformatted("simple node :)");
		ImNodes::EndNodeTitleBar();

		ImNodes::BeginInputAttribute(2);
		ImGui::Text("input");
		ImNodes::EndInputAttribute();

		ImNodes::BeginOutputAttribute(3);
		ImGui::Indent(40);
		ImGui::Text("output");
		ImNodes::EndOutputAttribute();

		ImNodes::EndNode();

		ImNodes::BeginNode(2);

		ImNodes::BeginNodeTitleBar();
		ImGui::TextUnformatted("simple node ):");
		ImNodes::EndNodeTitleBar();

		ImNodes::BeginInputAttribute(4);
		ImGui::Text("input");
		ImNodes::EndInputAttribute();

		ImNodes::BeginOutputAttribute(5);
		ImGui::Indent(40);
		ImGui::Text("output");
		ImNodes::EndOutputAttribute();

		ImNodes::EndNode();

		for (int i = 0; i < links.size(); ++i)
		{
			const std::pair<int, int> p = links[i];
			// in this case, we just use the array index of the link
			// as the unique identifier
			ImNodes::Link(i, p.first, p.second);
		}

		ImNodes::MiniMap();
		ImNodes::EndNodeEditor();

		int link;
		if (ImNodes::IsLinkHovered(&link))
		{
			if (ImGui::IsMouseClicked(ImGuiMouseButton_Right))
			{
				links.erase(links.begin() + link);
			}
		}

		int start_attr, end_attr;
		if (ImNodes::IsLinkCreated(&start_attr, &end_attr))
		{
			links.push_back(std::make_pair(start_attr, end_attr));
		}

		ImGui::End();
	}

  private:
	std::vector<std::pair<int, int>> links;
};
}        // namespace Ilum::panel

int main()
{
	auto image = Ilum::Resource::ResourceCache::LoadTexture2D("D:/Workspace/IlumEngine/Asset/Texture/Material/cavern-deposits/cavern-deposits_albedo.png");

	Ilum::Core::JobSystem::Initialize();

	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/cavern-deposits/cavern-deposits_ao.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/cavern-deposits/cavern-deposits_height.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/cavern-deposits/cavern-deposits_metallic.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/cavern-deposits/cavern-deposits_normal-ogl.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/cavern-deposits/cavern-deposits_roughness.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/dusty-cobble/dusty-cobble_albedo.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/dusty-cobble/dusty-cobble_ao.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/dusty-cobble/dusty-cobble_height.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/dusty-cobble/dusty-cobble_metallic.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/dusty-cobble/dusty-cobble_normal-ogl.png");
	Ilum::Resource::ResourceCache::LoadTexture2DAsync("D:/Workspace/IlumEngine/Asset/Texture/Material/dusty-cobble/dusty-cobble_roughness.png");

	Ilum::Resource::ResourceCache::OnUpdate();

	Ilum::Engine engine;

	Ilum::Editor::instance()->addPanel(std::make_unique<Ilum::panel::RGEditor>());

	Ilum::Graphics::RenderContext::GetWindow().SetIcon(std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/logo.bmp");

	while (!Ilum::Graphics::RenderContext::GetWindow().ShouldClose())
	{
		Ilum::Resource::ResourceCache::OnUpdate();

		engine.onTick();

		Ilum::Graphics::RenderContext::GetWindow().SetTitle((Ilum::Scene::instance()->name.empty() ? "IlumEngine" : Ilum::Scene::instance()->name));
	}

	return 0;
}