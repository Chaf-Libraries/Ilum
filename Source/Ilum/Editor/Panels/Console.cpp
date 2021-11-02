#include "Console.hpp"

#include "Logging/Logger.hpp"

#include "Loader/ImageLoader/ImageLoader.hpp"

#include "ImGui/ImGuiContext.hpp"

#include "Renderer/Renderer.hpp"

namespace Ilum::panel
{
Console::Console()
{
	m_name = "Console";

	ImageLoader::loadImageFromFile(m_icons[0], std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/trace.png");
	ImageLoader::loadImageFromFile(m_icons[1], std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/debug.png");
	ImageLoader::loadImageFromFile(m_icons[2], std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/info.png");
	ImageLoader::loadImageFromFile(m_icons[3], std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/warn.png");
	ImageLoader::loadImageFromFile(m_icons[4], std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/error.png");
	ImageLoader::loadImageFromFile(m_icons[5], std::string(PROJECT_SOURCE_DIR) + "Asset/Texture/Icon/critical.png");
}

void Console::draw(float delta_time)
{
	ImGui::Begin("Console", &active);

	static const char *ASSET_TYPE[] = {
	    "Engine",
	    "Vulkan",
	};
	static int current_item = 0;

	// 0-trace, 1-debug, 2 - info, 3 - warn, 4 - err, 5 - critical
	static bool   enable[6] = {true, true, true, true, true, true};
	static ImVec4 color[6]  = {
        ImVec4{0.f, 1.f, 1.f, 0.75f},
        ImVec4{1.f, 0.f, 1.f, 0.75f},
        ImVec4{0.f, 1.f, 0.f, 0.75f},
        ImVec4{1.f, 1.f, 0.f, 0.75f},
        ImVec4{1.f, 0.f, 0.f, 0.75f},
        ImVec4{0.f, 0.f, 1.f, 0.75f},
    };

	ImGui::Combo("Logging", &current_item, ASSET_TYPE, 2);
	m_filter.Draw("Filter");

	ImGui::SameLine();
	if (ImGui::Button("Clear"))
	{
		Logger::getInstance().clear();
	}

	ImGui::SameLine();
	if (ImGui::Button("Save"))
	{
		Logger::getInstance().save();
	}

	for (uint32_t i = 0; i < 6; i++)
	{
		if (ImGui::ImageButton(ImGuiContext::textureID(m_icons[i], Renderer::instance()->getSampler(Renderer::SamplerType::Trilinear_Clamp)),
		                       ImVec2{20.f, 20.f}, ImVec2{0, 0}, ImVec2{1, 1}, -1, ImVec4(0, 0, 0, 0), enable[i] ? ImVec4(1, 1, 1, 1) : ImVec4(0.6f, 0.6f, 0.6f, 0.6f)))
		{
			enable[i] = !enable[i];
		}

		if (i < 5)
		{
			ImGui::SameLine();
		}
	}

	ImGui::Separator();
	ImGui::BeginChild("scrolling", ImVec2(0, 0), false, ImGuiWindowFlags_HorizontalScrollbar);

	auto &logs = current_item == 0 ? Logger::getInstance().getLogs("engine") : Logger::getInstance().getLogs("vulkan");
	for (auto &log : logs)
	{
		if (!enable[log.level])
		{
			continue;
		}

		if (!m_filter.IsActive() || (m_filter.IsActive() && m_filter.PassFilter(log.msg.data(), log.msg.data() + log.msg.size())))
		{
			ImGui::TextColored(color[log.level], log.msg.c_str());
		}
	}

	if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY())
	{
		ImGui::SetScrollHereY(1.0f);
	}

	ImGui::EndChild();
	ImGui::End();
}
}        // namespace Ilum::panel