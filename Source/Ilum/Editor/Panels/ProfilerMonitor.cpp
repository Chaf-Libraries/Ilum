#include "ProfilerMonitor.hpp"

#include "Graphics/GraphicsContext.hpp"
#include "Graphics/Profiler.hpp"

#include <imgui.h>

namespace Ilum::panel
{
ProfilerMonitor::ProfilerMonitor()
{
	m_name = "Profiler";
	m_stopwatch.start();
}

void ProfilerMonitor::draw(float delta_time)
{
	ImGui::Begin("Profiler", &active);

	if (m_stopwatch.elapsedSecond() > 0.5f)
	{
		m_profile_result = GraphicsContext::instance()->getProfiler().getResult();
		m_stopwatch.start();
	}

	for (auto &[name, res] : m_profile_result)
	{
		auto [cpu_time, gpu_time] = res;

		ImGui::Text("%s: CPU time: %.3f ms, GPU time: %.3f ms", name.c_str(), cpu_time, gpu_time);
	}

	ImGui::End();
}
}        // namespace Ilum::panel