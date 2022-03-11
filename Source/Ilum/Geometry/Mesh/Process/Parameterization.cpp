#include "Parameterization.hpp"

#include "Geometry/Mesh/HEMesh.hpp"

#include <glm/gtc/constants.hpp>

__pragma(warning(push, 0))
#include <Eigen/Eigen>
    __pragma(warning(pop))

        namespace Ilum::geometry
{
	std::pair<std::vector<Vertex>, std::vector<uint32_t>> Parameterization::MinimumSurface(const std::vector<Vertex> &in_vertices, const std::vector<uint32_t> &in_indices)
	{
		HEMesh hemesh(preprocess(in_vertices), in_indices);

		auto boundary = std::move(hemesh.longestBoundary());

		// Build Laplace Matrix
		size_t nV = hemesh.vertices().size();

		std::vector<Eigen::Triplet<float>> Lij;

		for (size_t i = 0; i < nV; i++)
		{
			auto *v = hemesh.vertices()[i];
			Lij.push_back(Eigen::Triplet<float>(static_cast<int32_t>(i), static_cast<int32_t>(i), 1.f));
			if (std::find(boundary.begin(), boundary.end(), v) == boundary.end())
			{
				auto adj_vertices = hemesh.adjVertices(v);
				for (size_t j = 0; j < adj_vertices.size(); j++)
				{
					Lij.push_back(Eigen::Triplet<float>(static_cast<int32_t>(i), static_cast<int32_t>(hemesh.vertexIndex(adj_vertices[j])), -1.f / static_cast<float>(adj_vertices.size())));
				}
			}
		}

		Eigen::SparseMatrix<float> Laplace_matrix;
		Laplace_matrix.resize(nV, nV);
		Laplace_matrix.setZero();
		Laplace_matrix.setFromTriplets(Lij.begin(), Lij.end());

		// LU solver
		Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;

		solver.compute(Laplace_matrix);
		if (solver.info() != Eigen::Success)
		{
			LOG_ERROR("Laplace Matrix Is Error!");
			return std::make_pair(in_vertices, in_indices);
		}

		// Setting end condition
		Eigen::MatrixXf V(nV, 3);
		Eigen::MatrixXf b(nV, 3);

		V.setZero();
		b.setZero();

		for (size_t i = 0; i < nV; i++)
		{
			auto *v = hemesh.vertices()[i];
			if (std::find(boundary.begin(), boundary.end(), v) != boundary.end())
			{
				b(i, 0) = v->position.x;
				b(i, 1) = v->position.y;
				b(i, 2) = v->position.z;
			}
		}

		// Solve
		V = solver.solve(b);

		for (size_t i = 0; i < nV; i++)
		{
			auto *v       = hemesh.vertices()[i];
			v->position.x = V(i, 0);
			v->position.y = V(i, 1);
			v->position.z = V(i, 2);
		}

		auto [vertices, indices] = hemesh.toMesh();

		std::vector<glm::vec2> texcoords(in_vertices.size());
		for (size_t i = 0; i < texcoords.size(); i++)
		{
			texcoords[i] = in_vertices[i].texcoord;
		}

		return std::make_pair(postprocess(vertices, indices, texcoords), std::move(indices));
	}

	std::pair<std::vector<Vertex>, std::vector<uint32_t>> Parameterization::TutteParameterization(const std::vector<Vertex> &in_vertices, const std::vector<uint32_t> &in_indices, TutteWeightType weight_type, TutteBorderType border_type)
	{
		HEMesh hemesh(preprocess(in_vertices), in_indices);
		auto   boundary = std::move(hemesh.longestBoundary());

		// Build Laplace Matrix
		size_t nV = hemesh.vertices().size();

		std::vector<Eigen::Triplet<float>> Lij;

		for (size_t i = 0; i < nV; i++)
		{
			auto *v = hemesh.vertices()[i];
			Lij.push_back(Eigen::Triplet<float>(static_cast<int32_t>(i), static_cast<int32_t>(i), 1.f));
			if (std::find(boundary.begin(), boundary.end(), v) == boundary.end())
			{
				auto adj_vertices = hemesh.adjVertices(v);

				if (weight_type == TutteWeightType::Uniform)
				{
					for (size_t j = 0; j < adj_vertices.size(); j++)
					{
						Lij.push_back(Eigen::Triplet<float>(static_cast<int32_t>(i), static_cast<int32_t>(hemesh.vertexIndex(adj_vertices[j])), -1.f / static_cast<float>(adj_vertices.size())));
					}
				}
				else if (weight_type == TutteWeightType::Cotangent)
				{
					std::unordered_map<std::pair<int32_t, int32_t>, float, pair_hash> weights;

					float sum_weight = 0.f;

					for (size_t j = 0; j < adj_vertices.size(); j++)
					{
						auto *current = adj_vertices[j];
						auto *next    = adj_vertices[(j + 1) % adj_vertices.size()];
						auto *prev    = adj_vertices[(j + adj_vertices.size() - 1) % adj_vertices.size()];

						float cos1 = glm::dot(glm::normalize(v->position - next->position), glm::normalize(current->position - next->position));
						float cos2  = glm::dot(glm::normalize(v->position - prev->position), glm::normalize(current->position - prev->position));

						float alpha1 = acosf(cos1);
						float alpha2 = acosf(cos2);

						// cot_alpha+cot_beta
						float weight = 1.f / tanf(alpha1) + 1.f / tanf(alpha2);

						weights[std::make_pair(static_cast<int32_t>(i), static_cast<int32_t>(hemesh.vertexIndex(current)))] = weight;
						sum_weight += weight;
					}

					for (auto &[pair, weight] : weights)
					{
						Lij.push_back(Eigen::Triplet<float>(pair.first, pair.second, -weight / sum_weight));
					}
				}
			}
		}

		Eigen::SparseMatrix<float> Laplace_matrix;
		Laplace_matrix.resize(nV, nV);
		Laplace_matrix.setZero();
		Laplace_matrix.setFromTriplets(Lij.begin(), Lij.end());

		// LU solver
		Eigen::SparseLU<Eigen::SparseMatrix<float>> solver;

		solver.compute(Laplace_matrix);
		if (solver.info() != Eigen::Success)
		{
			LOG_ERROR("Laplace Matrix Is Error!");
			return std::make_pair(in_vertices, in_indices);
		}

		// Set end condition
		Eigen::MatrixXf B(nV, 2);
		Eigen::MatrixXf X(nV, 2);

		B.setZero();
		X.setZero();

		std::vector<glm::vec2> boundary_list;
		size_t                 nB = boundary.size();

		for (uint32_t i = 0; i < nB; i++)
		{
			if (border_type == TutteBorderType::Circle)
			{
				float theta = 2.f * glm::pi<float>() / static_cast<float>(nB);
				boundary_list.push_back(glm::vec2(0.5f * cos(static_cast<float>(i) * theta) + 0.5f, 0.5f * sin(static_cast<float>(i) * theta) + 0.5f));
			}
			else if (border_type == TutteBorderType::Rectangle)
			{
				float step = 4.f / static_cast<float>(nB);
				float temp = static_cast<float>(i) * step;
				if (temp < 1.f)
				{
					boundary_list.push_back(glm::vec2(0.f, temp));
				}
				else if (temp < 2.f && temp >= 1.f)
				{
					boundary_list.push_back(glm::vec2(temp - 1.f, 1.f));
				}
				else if (temp < 3.f && temp >= 2.f)
				{
					boundary_list.push_back(glm::vec2(1.f, 3.f - temp));
				}
				else
				{
					boundary_list.push_back(glm::vec2(4 - temp, 0));
				}
			}
		}

		for (int i = 0; i < boundary_list.size(); i++)
		{
			auto *v                     = boundary[i];
			B(hemesh.vertexIndex(v), 0) = boundary_list[i].x;
			B(hemesh.vertexIndex(v), 1) = boundary_list[i].y;
		}

		// Solve
		X = solver.solve(B);

		std::vector<glm::vec2> texcoords(nV);
		for (size_t i = 0; i < nV; i++)
		{
			texcoords[i].x = X(i, 0);
			texcoords[i].y = X(i, 1);
		}

		auto [vertices, indices] = hemesh.toMesh();

		return std::make_pair(postprocess(vertices, indices, texcoords), std::move(indices));
	}
}        // namespace Ilum::geometry