#include "LightUpdate.hpp"

#include "Scene/Component/DirectionalLight.hpp"
#include "Scene/Component/Light.hpp"
#include "Scene/Component/PointLight.hpp"
#include "Scene/Component/SpotLight.hpp"
#include "Scene/Entity.hpp"
#include "Scene/Scene.hpp"

#include <tbb/tbb.h>

namespace Ilum::sym
{
void LightUpdate::run()
{
	auto view = Scene::instance()->getRegistry().view<cmpt::Light>();
	tbb::parallel_for_each(view.begin(), view.end(), [&view](auto entity) {
		auto &light = Entity(entity).getComponent<cmpt::Light>();
		if (!light.impl || light.impl->type() != light.type)
		{
			switch (light.type)
			{
				case cmpt::LightType::Directional:
					light.impl = createScope<cmpt::DirectionalLight>();
					break;
				case cmpt::LightType::Point:
					light.impl = createScope<cmpt::PointLight>();
					break;
				case cmpt::LightType::Spot:
					light.impl = createScope<cmpt::SpotLight>();
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