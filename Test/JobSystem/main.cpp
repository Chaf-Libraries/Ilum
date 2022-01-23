#include <Core/JobSystem/JobSystem.hpp>

#include <array>
#include <iostream>

using namespace Ilum::Core;

std::atomic<uint32_t> count;

std::thread::id test_func()
{
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	return std::this_thread::get_id();
}

int main()
{
	JobSystem::Initialize();

	{
		// Job Graph:
		// For parallel tasks with dependencies
		// 1 -> 5
		// 1 -> 6
		// 2 -> 4
		// 2 -> 5
		// 3 -> 6
		// 4 -> 5
		// 6 -> 5
		// 5 -> 7
		// 5 -> 8
		// 5 -> 9
		// 5 -> 10

		std::cout << "Job Graph Test" << std::endl;

		// Declare a job graph
		JobGraph job_graph;
		JobGraph sub_graph;

		std::array<std::thread::id, 10> thread_ids;

		// Declare some nodes
		JobNode node1([&thread_ids]() { 
			thread_ids[0] = test_func(); });
		JobNode node2([&thread_ids]() { thread_ids[1] = test_func(); });
		JobNode node3([&thread_ids]() { thread_ids[2] = test_func(); });
		JobNode node4([&thread_ids]() { thread_ids[3] = test_func(); });
		JobNode node5([&thread_ids]() { thread_ids[4] = test_func(); });
		JobNode node6([&thread_ids]() { thread_ids[5] = test_func(); });
		JobNode node7([&thread_ids]() { thread_ids[6] = test_func(); });
		JobNode node8([&thread_ids]() { thread_ids[7] = test_func(); });
		JobNode node9([&thread_ids]() { thread_ids[8] = test_func(); });
		JobNode node10([&thread_ids]() { 
			thread_ids[9] = test_func(); });

		// Adding nodes to the graph
		job_graph.addNode(&node1)
		    .addNode(&node2)
		    .addNode(&node3)
		    .addNode(&node4)
		    .addNode(&node5)
		    .addNode(&node6)
		    .addNode(&node7)
			.addNode(&sub_graph);

		sub_graph.addNode(&node8)
		    .addNode(&node9)
		    .addNode(&node10);

		// Setting topology
		node1.Percede(&node5);
		node1.Percede(&node6);
		node2.Percede(&node4);
		node2.Percede(&node5);
		node3.Percede(&node6);
		node4.Percede(&node5);
		node6.Percede(&node5);
		node5.Percede(&node7);
		node5.Percede(&sub_graph);
		node8.Percede(&node9);
		node8.Percede(&node10);

		// Compilation will check DAG and topology sort
		bool result = job_graph.Compile();

		// Run the job graph
		JobHandle job_graph_handle;
		JobSystem::Execute(job_graph_handle, job_graph);

		// Wait for the result
		JobSystem::Wait(job_graph_handle);

		for (uint32_t i = 0; i < 10; i++)
		{
			std::cout << "node " << i << " on thread " << thread_ids[i] << std::endl;
		}
	}

	{
		// Future Pattern:
		// For tasks run in the background, like async I/O

		std::cout << "Future Pattern Test" << std::endl;

		JobHandle future_handle;

		// Store the results
		std::vector<std::future<std::thread::id>> futures;

		for (uint32_t i = 0; i < 10; i++)
		{
			// Adding tasks, saving result
			futures.emplace_back(JobSystem::Execute(future_handle, test_func));
		}

		// Waiting for the result, you can also do other things
		while (!futures.empty())
		{
			for (auto iter = futures.begin(); iter != futures.end();)
			{
				// Get the result
				if (iter->wait_for(std::chrono::seconds(0)) == std::future_status::ready)
				{
					std::cout << "Run on thread " << iter->get() << std::endl;
					iter = futures.erase(iter);
				}
				else
				{
					iter++;
				}
			}
		}
	}

	{
		// Dispatch Test:
		// Like compute shader, it can be used for image processing or ray tracing
		std::cout << "Dispatch Test" << std::endl;

		JobHandle dispatch_handle;

		// We will fill a 10 length array
		uint32_t data[100];

		JobSystem::Dispatch(dispatch_handle, 100, 10, [&data](uint32_t group_id) {
			for (uint32_t i = 0; i < 10; i++)
			{
				data[group_id * 10 + i] = group_id;
			}
		});

		JobSystem::Wait(dispatch_handle);

		for (uint32_t i = 0; i < 100; i++)
		{
			std::cout << data[i] << " ";
		}
	}

	return 0;
}