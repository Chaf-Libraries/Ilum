#include "Parameterization.hpp"

#include "Geometry/Mesh/HEMesh.hpp"

__pragma(warning(push, 0))
#include <Eigen/Eigen>

    __pragma(warning(pop))

        namespace Ilum::geometry
{
	std::pair<std::vector<Vertex>, std::vector<uint32_t>> Parameterization::MinimumSurface(const std::vector<Vertex> &in_vertices, const std::vector<uint32_t> &in_indices)
	{
		HEMesh hemesh(preprocess(in_vertices), in_indices);

		size_t longest_boundaries = 0;
		auto   boundaries         = hemesh.boundary();

		if (boundaries.empty())
		{
			LOG_ERROR("Mesh doesn't have boundary");
			return std::make_pair(in_vertices, in_indices);
		}

		// Find longest boundary
		for (size_t i = 0; i < boundaries.size(); i++)
		{
			if (boundaries[longest_boundaries].size() < boundaries[i].size())
			{
				longest_boundaries = i;
			}
		}
		auto boundary = std::move(boundaries[longest_boundaries]);

		// Build Laplace Matrix
		size_t nV = hemesh.vertices().size();

		std::vector<Eigen::Triplet<float>> Lij;

		for (size_t i = 0; i < nV; i++)
		{
			auto *v = hemesh.vertices()[i];
			Lij.push_back(Eigen::Triplet<float>(static_cast<int32_t>(i), static_cast<int32_t>(i), 1.f));
			if (std::find(boundary.begin(), boundary.end(), v) == boundary.end())
			//if (!hemesh.onBoundary(v))
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

		Eigen::MatrixXf V(nV, 3);
		Eigen::MatrixXf b(nV, 3);

		V.setZero();
		b.setZero();

		for (size_t i = 0; i < nV; i++)
		{
			auto *v = hemesh.vertices()[i];
			//if (hemesh.onBoundary(v))
			if (std::find(boundary.begin(), boundary.end(), v) != boundary.end())
			{
				b(i, 0) = v->position.x;
				b(i, 1) = v->position.y;
				b(i, 2) = v->position.z;
			}
		}

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

	std::pair<std::vector<Vertex>, std::vector<uint32_t>> Parameterization::TutteParameterization(const std::vector<Vertex> &in_vertices, const std::vector<uint32_t> &in_indices, TutteWeightType weight_type)
	{
		return std::pair<std::vector<Vertex>, std::vector<uint32_t>>();
	}
}        // namespace Ilum::geometry