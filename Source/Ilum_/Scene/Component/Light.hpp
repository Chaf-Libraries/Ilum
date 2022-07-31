#pragma once

#include "Component.hpp"

#include <RHI/Buffer.hpp>

#include <entt.hpp>

#include <imgui.h>
#include <imgui_internal.h>

namespace Ilum::cmpt
{
enum class LightType
{
	Point,
	Directional,
	Spot,
	Area
};

enum class AreaLightShape
{
	Rectangle,
	Ellipse
};

class Light : public Component
{
  public:
	Light() = default;

	bool OnImGui(ImGuiContext &context);

	virtual void Tick(Scene &scene, entt::entity entity, RHIDevice *device) override;

	Texture *GetShadowmap();
	Buffer  *GetBuffer();

	void SetType(LightType type);

	LightType GetType() const;

	template <class Archive>
	void serialize(Archive &ar)
	{
		ar(m_type, m_color, m_intensity, m_range, m_spot_inner_cone_angle, m_spot_outer_cone_angle);
	}

  private:
	void UpdateDirectionalLight(Scene &scene, entt::entity entity, RHIDevice *device);
	void UpdatePointLight(Scene &scene, entt::entity entity, RHIDevice *device);
	void UpdateSpotLight(Scene &scene, entt::entity entity, RHIDevice *device);
	void UpdateAreaLight(Scene &scene, entt::entity entity, RHIDevice *device);

  private:
	LightType m_type = LightType::Point;

	glm::vec3 m_color     = {1.f, 1.f, 1.f};
	float     m_intensity = 1.f;

	float m_range                 = 1.f;
	float m_radius                = 1.f;
	float m_spot_inner_cone_angle = 12.5f;
	float m_spot_outer_cone_angle = 17.5f;

	bool m_cast_shadow = false;

	AreaLightShape m_shape = AreaLightShape::Rectangle;

	// Shadow map
	std::unique_ptr<Texture> m_shadow_map = nullptr;
	std::unique_ptr<Buffer>  m_buffer     = nullptr;
};
}        // namespace Ilum::cmpt