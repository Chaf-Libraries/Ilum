#include <iostream>
#include <RenderGraph/RenderGraphBlackboard.hpp>
#include <RHI/RHIBuffer.hpp>

using namespace Ilum;

struct TestStruct
{
	~TestStruct()
	{
		std::cout << "Fuck you";
	}
};

struct LightData
{
	std::unique_ptr<RHIBuffer> directional_lights;
	std::unique_ptr<RHIBuffer> point_lights;
	int                        a;
	float                      b;
	TestStruct                 test;
};

int main()
{
	RenderGraphBlackboard black_board;
	auto                 *light_buffer = black_board.Get<LightData>();
	light_buffer->a                    = 1;
	light_buffer->b                    = 0.2f;

	return 0;
}