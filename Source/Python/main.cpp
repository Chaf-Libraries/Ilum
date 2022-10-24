#include <Core/Window.hpp>
#include <RHI/RHIContext.hpp>
#include <RHI/RHIDefinitions.hpp>
#include <RHI/RHIDevice.hpp>

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace Ilum
{
PYBIND11_MODULE(Ilum, m)
{
	py::class_<Window>(m, "Window")
	    .def(py::init<const std::string &, const std::string &, uint32_t, uint32_t>())
	    .def("Tick", &Window::Tick)
	    .def("IsKeyDown", &Window::IsKeyDown)
	    .def("IsMouseButtonDown", &Window::IsMouseButtonDown)
	    .def("SetTitle", &Window::SetTitle)
	    .def("GetWidth", &Window::GetWidth)
	    .def("GetHeight", &Window::GetHeight);

	py::class_<RHIContext>(m, "RHIContext")
	    .def(py::init<Window *, bool>())
	    .def("GetDeviceName", &RHIContext::GetDeviceName)
	    .def("GetBackend", &RHIContext::GetBackend)
	    .def("IsVsync", &RHIContext::IsVsync)
	    .def("SetVsync", &RHIContext::SetVsync)
	    .def("IsFeatureSupport", &RHIContext::IsFeatureSupport)
	    .def("WaitIdle", &RHIContext::WaitIdle)
	    .def("GetSwapchain", &RHIContext::GetSwapchain)
	    .def("CreateSwapchain", &RHIContext::CreateSwapchain)
	    .def("CreateTexture", &RHIContext::CreateTexture)
	    .def("CreateTexture2D", &RHIContext::CreateTexture2D)
	    .def("CreateTexture3D", &RHIContext::CreateTexture3D)
	    .def("CreateTextureCube", &RHIContext::CreateTextureCube)
	    .def("CreateTexture2DArray", &RHIContext::CreateTexture2DArray)
	    .def("MapToCUDATexture", &RHIContext::MapToCUDATexture)
	    .def("CreateBuffer", static_cast<std::unique_ptr<RHIBuffer> (RHIContext::*)(const BufferDesc &)>(&RHIContext::CreateBuffer))
	    .def("CreateBuffer", static_cast<std::unique_ptr<RHIBuffer> (RHIContext::*)(size_t, RHIBufferUsage, RHIMemoryUsage)>(&RHIContext::CreateBuffer))
	    .def("CreateSampler", &RHIContext::CreateSampler)
	    .def("CreateCommand", &RHIContext::CreateCommand)
	    .def("CreateDescriptor", &RHIContext::CreateDescriptor)
	    .def("CreatePipelineState", &RHIContext::CreatePipelineState)
	    .def("CreateShader", &RHIContext::CreateShader)
	    .def("CreateRenderTarget", &RHIContext::CreateRenderTarget)
	    .def("CreateProfiler", &RHIContext::CreateProfiler)
	    .def("CreateFence", &RHIContext::CreateFence)
	    .def("CreateFrameFence", &RHIContext::CreateFrameFence)
	    .def("CreateSemaphore", &RHIContext::CreateSemaphore)
	    .def("CreateFrameSemaphore", &RHIContext::CreateFrameSemaphore)
	    .def("MapToCUDASemaphore", &RHIContext::MapToCUDASemaphore)
	    .def("CreateAcccelerationStructure", &RHIContext::CreateAcccelerationStructure)
	    //.def("Submit", &RHIContext::Submit)
	    .def("Execute", static_cast<void (RHIContext::*)(RHICommand *)>(&RHIContext::Execute))
	    //.def("Execute", static_cast<void (RHIContext::*)(std::vector<RHICommand *> &&, std::vector<RHISemaphore *> &&, std::vector<RHISemaphore *> &&, RHIFence *)>(&RHIContext::Execute))
	    .def("Reset", &RHIContext::Reset)
	    .def("GetBackBuffer", &RHIContext::GetBackBuffer)
	    .def("BeginFrame", &RHIContext::BeginFrame)
	    .def("EndFrame", &RHIContext::EndFrame);

	py::enum_<RHIBackend>(m, "Backend", py::arithmetic())
	    .value("Vulkan", RHIBackend::Vulkan)
	    .export_values();

	class PyDevice : public RHIDevice
	{
	  public:
		using RHIDevice::RHIDevice;

		virtual void WaitIdle() override
		{
			PYBIND11_OVERRIDE_PURE(
			    void,
			    RHIDevice,
			    WaitIdle);
		}

		virtual bool IsFeatureSupport(RHIFeature feature) override
		{
			PYBIND11_OVERRIDE_PURE(
			    bool,
			    RHIDevice,
			    IsFeatureSupport,
			    feature);
		}
	};

	py::class_<RHIDevice, PyDevice /* <--- trampoline*/>(m, "Device")
	    .def(py::init<RHIBackend>())
	    .def("GetName", &RHIDevice::GetName)
	    .def("GetBackend", &RHIDevice::GetBackend)
	    .def("WaitIdle", &RHIDevice::WaitIdle)
	    .def("IsFeatureSupport", &RHIDevice::IsFeatureSupport)
	    .def("Create", &RHIDevice::Create);
}
}        // namespace Ilum