#pragma once

#include "Utils/PCH.hpp"

namespace Ilum
{
namespace Graphics
{
class Buffer;
};

class CommandBuffer;

class Profiler
{
  public:
	struct Sample
	{
		std::string name    = "";
		bool        has_gpu = false;

		struct Data
		{
			uint32_t                                                    index = 0;
			std::chrono::time_point<std::chrono::high_resolution_clock> cpu_time;
		};

		Data start;
		Data end;
	};

  public:
	Profiler();

	~Profiler();

	void beginFrame(const CommandBuffer &cmd_buffer);

	void beginSample(const std::string &name, const CommandBuffer &cmd_buffer);

	void beginSample(const std::string &name);

	void endSample(const std::string &name, const CommandBuffer &cmd_buffer);

	void endSample(const std::string &name);

	// name - [cpu, gpu]
	std::map<std::string, std::pair<float, float>> getResult() const;

  private:
	std::vector<VkQueryPool>      m_query_pools;
	std::vector<Graphics::Buffer> m_buffers;
	uint32_t                      m_current_index = 0;

	std::vector<std::map<std::string, Sample>> m_samples;
};
}        // namespace Ilum