#include "Profiler.hpp"
#include "Device.hpp"
#include "Command.hpp"

namespace Ilum::CUDA
{
Profiler::Profiler(RHIDevice *device, uint32_t frame_count) :
    RHIProfiler(device, frame_count)
{
	m_start_events.resize(frame_count);
	m_end_events.resize(frame_count);

	for (uint32_t i = 0; i < frame_count; i++)
	{
		cudaEventCreate(&m_start_events[i]);
		cudaEventCreate(&m_end_events[i]);
	}
}

Profiler::~Profiler()
{
	for (uint32_t i = 0; i < m_frame_count; i++)
	{
		cudaEventDestroy(m_start_events[i]);
		cudaEventDestroy(m_end_events[i]);
	}
}

void Profiler::Begin(RHICommand *cmd_buffer, uint32_t frame_index)
{
	m_current_index = frame_index;

	m_state.thread_id = std::this_thread::get_id();

	cudaEventSynchronize(m_end_events[m_current_index]);
	cudaEventElapsedTime(&m_state.gpu_time, m_start_events[m_current_index], m_end_events[m_current_index]);

	static_cast<Command *>(cmd_buffer)->EventRecord(m_start_events[m_current_index]);
	m_state.cpu_time = std::chrono::duration<float, std::milli>(m_state.cpu_end - m_state.cpu_start).count();

	m_state.cpu_start = std::chrono::high_resolution_clock::now();
}

void Profiler::End(RHICommand *cmd_buffer)
{
	static_cast<Command *>(cmd_buffer)->EventRecord(m_end_events[m_current_index]);

	m_state.cpu_end = std::chrono::high_resolution_clock::now();
}
}        // namespace Ilum::CUDA