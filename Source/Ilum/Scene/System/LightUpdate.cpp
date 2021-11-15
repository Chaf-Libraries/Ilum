#include "LightUpdate.hpp"

#include "Scene/Component/Light.hpp"
#include "Scene/Component/DirectionalLight.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

namespace Ilum::sym
{
void LightUpdate::run()
{
	auto view = Scene::instance()->getRegistry().view<cmpt::Light>();
	std::for_each(std::execution::par_unseq, view.begin(), view.end(), [&view](auto entity) {
		auto &light = Entity(entity).getComponent<cmpt::Light>();
		if (!light.impl||light.impl->type()!=light.type)
		{
			switch (light.type)
			{
				case cmpt::LightType::Directional:
					light.impl = createScope<cmpt::DirectionalLight>();
					break;
				case cmpt::LightType::Point:

					break;
				case cmpt::LightType::Spot:

					break;
				case cmpt::LightType::Area:

					break;
				default:
					break;
			}
		}
	});
}
}        // namespace Ilum::sym