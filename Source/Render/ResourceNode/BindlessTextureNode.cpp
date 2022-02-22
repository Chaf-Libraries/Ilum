#pragma once

#include "BindlessTextureNode.hpp"
#include "RenderGraph/PassNode.hpp"

#include <Resource/ResourceCache.hpp>

#include <imnodes.h>

namespace Ilum::Render
{
inline void draw_texture_asset(float height, float space)
{
	auto &image_cache = Resource::ResourceCache::GetTexture2DQuery();

	float width = 0.f;

	ImGuiStyle &style       = ImGui::GetStyle();
	style.ItemSpacing       = ImVec2(10.f, 10.f);
	float window_visible_x2 = ImGui::GetWindowPos().x + ImGui::GetWindowContentRegionMax().x;

	for (auto &[name, index] : image_cache)
	{
		auto image = Resource::ResourceCache::LoadTexture2D(name);

		ImGui::Button(name.c_str(), {height / static_cast<float>(image.get().GetHeight()) * static_cast<float>(image.get().GetWidth()), height});

		// Drag&Drop source
		if (ImGui::BeginDragDropSource())
		{
			if (image.get().GetLayerCount() == 1)
			{
				ImGui::SetDragDropPayload("Texture2D", &name, sizeof(std::string));
			}
			else if (image.get().GetLayerCount() == 6)
			{
				ImGui::SetDragDropPayload("TextureCube", &name, sizeof(std::string));
				ImGui::SetDragDropPayload("TextureArray", &name, sizeof(std::string));
			}
			else
			{
				ImGui::SetDragDropPayload("TextureArray", &name, sizeof(std::string));
			}
			ImGui::EndDragDropSource();
		}

		// Graphics::Image Hint
		if (ImGui::BeginPopupContextItem(name.c_str()))
		{
			if (ImGui::MenuItem("Delete"))
			{
				Resource::ResourceCache::RemoveTexture2D(name);
			}
			ImGui::EndPopup();
		}
		else if (ImGui::IsItemHovered() && ImGui::IsWindowFocused())
		{
			ImVec2 pos = ImGui::GetIO().MousePos;
			ImGui::SetNextWindowPos(ImVec2(pos.x + 10.f, pos.y + 10.f));
			ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_Tooltip | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar);
			ImGui::Text(name.c_str());
			ImGui::Separator();
			ImGui::Text("format: %s", std::to_string(image.get().GetFormat()).c_str());
			ImGui::Text("width: %s", std::to_string(image.get().GetWidth()).c_str());
			ImGui::Text("height: %s", std::to_string(image.get().GetHeight()).c_str());
			ImGui::Text("mip levels: %s", std::to_string(image.get().GetMipLevelCount()).c_str());
			ImGui::Text("layers: %s", std::to_string(image.get().GetLayerCount()).c_str());
			ImGui::End();
		}

		float last_button_x2 = ImGui::GetItemRectMax().x;
		float next_button_x2 = last_button_x2 + style.ItemSpacing.x + height / static_cast<float>(image.get().GetHeight()) * static_cast<float>(image.get().GetWidth());
		if (next_button_x2 < window_visible_x2)
		{
			ImGui::SameLine();
		}
	}
}

BindlessTextureNode::BindlessTextureNode(RenderGraph &render_graph) :
    IResourceNode("Bindless Texture Node", render_graph)
{
}

void BindlessTextureNode::OnImGui()
{

}

void BindlessTextureNode::OnImNode()
{
	const float node_width = 100.f;
	ImNodes::BeginNode(m_uuid);

	ImNodes::BeginNodeTitleBar();
	ImGui::TextUnformatted(m_name.c_str());
	ImNodes::EndNodeTitleBar();

	ImNodes::BeginOutputAttribute(m_read_id);
	const float label_width = ImGui::CalcTextSize("output").x;
	ImGui::Indent(node_width);
	ImGui::Text("output");
	ImNodes::EndOutputAttribute();

	ImNodes::EndNode();
}

void BindlessTextureNode::OnUpdate()
{
	if (Resource::ResourceCache::ImageUpdate())
	{
		m_images = Resource::ResourceCache::GetTexture2DReference();
		for (auto &[pass, pin] : m_read_passes)
		{
			pass->Bind(pin, m_images, AccessMode::Read);
		}
	}
}

bool BindlessTextureNode::_ReadBy(IPassNode *pass, int32_t pin)
{
	return pass->Bind(pin, m_images, AccessMode::Read);
	//return true;
}

bool BindlessTextureNode::_WriteBy(IPassNode *pass, int32_t pin)
{
	return false;
}
}        // namespace Ilum::Render