#include "MeshRenderer.hpp"
#include "Transform.hpp"

#include <RHI/Buffer.hpp>

#include <Asset/AssetManager.hpp>

#include "Scene/Component/Light.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

namespace Ilum::cmpt
{
void MeshRenderer::SetMesh(Mesh *mesh)
{
	m_mesh = mesh;
}

Buffer *MeshRenderer::GetBuffer()
{
	if (m_buffer)
	{
		return m_buffer.get();
	}
	return nullptr;
}

Mesh *MeshRenderer::GetMesh()
{
	return m_mesh;
}

void MeshRenderer::Tick(Scene &scene, entt::entity entity, RHIDevice *device)
{
	m_manager = &scene.GetAssetManager();

	if (m_update)
	{
		Entity e         = Entity(scene, entity);
		auto  &transform = e.GetComponent<cmpt::Transform>();

		if (!m_buffer)
		{
			BufferDesc desc   = {};
			desc.size         = sizeof(ShaderInterop::Instance);
			desc.buffer_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
			desc.memory_usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
			m_buffer          = std::make_unique<Buffer>(device, desc);
		}

		auto *instance_data          = static_cast<ShaderInterop::Instance *>(m_buffer->Map());
		instance_data->transform     = transform.GetWorldTransform();
		if (m_manager->IsValid(m_mesh))
		{
			instance_data->material      = m_manager->GetIndex(m_mesh->GetMaterial());
			instance_data->mesh          = m_manager->GetIndex(m_mesh);
			instance_data->meshlet_count = m_mesh->GetMeshletsCount();
		}
		m_buffer->Flush(m_buffer->GetSize());
		m_buffer->Unmap();

		// Update shadowmaps
		auto view = scene.GetRegistry().view<cmpt::Light>();
		view.each([&](entt::entity entity, cmpt::Light &light) {
			light.Update();
		});

		m_update = false;
	}
}

bool MeshRenderer::OnImGui(ImGuiContext &context)
{
	if (m_mesh && m_manager && m_manager->IsValid(m_mesh))
	{
		if (m_mesh->OnImGui(context))
		{
			m_mesh->UpdateBuffer();
			m_update = true;
		}
	}
	return m_update;
}

}        // namespace Ilum::cmpt