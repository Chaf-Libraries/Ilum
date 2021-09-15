#include "GraphicsContext.hpp"

#include "Core/Device/Instance.hpp"
#include "Core/Device/LogicalDevice.hpp"
#include "Core/Device/PhysicalDevice.hpp"
#include "Core/Device/Surface.hpp"
#include "Core/Device/Window.hpp"
#include "Core/Engine/Context.hpp"

#include "RenderPass/Swapchain.hpp"

namespace Ilum
{
GraphicsContext::GraphicsContext(Context *context) :
    TSubsystem<GraphicsContext>(context),
    m_instance(createScope<Instance>()),
    m_physical_device(createScope<PhysicalDevice>(*m_instance)),
    m_surface(createScope<Surface>(*m_physical_device, m_context->getSubsystem<Window>()->getSDLHandle())),
    m_logical_device(createScope<LogicalDevice>(*m_surface))
{
}
}        // namespace Ilum