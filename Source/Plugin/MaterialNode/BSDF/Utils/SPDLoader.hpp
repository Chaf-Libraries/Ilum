#pragma once

#include <Material/Spectrum.hpp>

#include <glm/glm.hpp>

#include <fstream>

namespace Ilum
{
// Loading *.spd spectrum data
class SPDLoader
{
  public:
	inline static glm::vec3 Load(const std::string &path)
	{
		std::fstream fs;
		fs.open(path.c_str(), std::ios::in);

		std::vector<float> vals;

		std::string line;
		while (std::getline(fs, line))
		{
			if (!line.empty() && line[0] != '#')
			{
				auto line_data = Path::GetInstance().Split(line, ' ');
				for (auto &x : line_data)
				{
					vals.push_back(std::stof(x));
				}
			}
		}

		std::vector<float> wls, v;
		for (size_t j = 0; j < vals.size() / 2; ++j)
		{
			wls.push_back(vals[2 * j]);
			v.push_back(vals[2 * j + 1]);
		}
		glm::vec3 result = FromSampled(&wls[0], &v[0], static_cast<int32_t>(wls.size()));

		fs.close();

		return result;
	}
};
}        // namespace Ilum