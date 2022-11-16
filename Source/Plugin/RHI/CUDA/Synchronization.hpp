#pragma once

#include "RHI/RHISynchronization.hpp"

#include <cuda.h>
#include <cuda_runtime.h>

#ifdef _WIN64
#	include <Windows.h>
#endif        // _WIN64

namespace Ilum::CUDA
{
class Semaphore : public RHISemaphore
{
  public:
	Semaphore(RHIDevice *device, HANDLE handle);

	~Semaphore();

	virtual void SetName(const std::string &name) override;

	void Signal();

	void Wait();

  private:
	cudaExternalSemaphore_t m_handle = nullptr;
};
}        // namespace Ilum::CUDA