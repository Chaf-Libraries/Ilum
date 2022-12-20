#include <Resource/Importer.hpp>
#include <Resource/Resource/Animation.hpp>
#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/Prefab.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/ResourceManager.hpp>

#include <Geometry/Meshlet.hpp>

#include <Animation/Animation.hpp>

#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <meshoptimizer.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

using namespace Ilum;

using Vertex        = Resource<ResourceType::Mesh>::Vertex;
using SkinnedVertex = Resource<ResourceType::SkinnedMesh>::SkinnedVertex;
using Node          = Resource<ResourceType::Prefab>::Node;

inline uint32_t PackTriangle(uint8_t v0, uint8_t v1, uint8_t v2)
{
	return static_cast<uint32_t>(v0) + (static_cast<uint32_t>(v1) << 8) + (static_cast<uint32_t>(v2) << 16);
}

inline glm::mat4 ToMatrix(const aiMatrix4x4 &matrix)
{
	return glm::mat4(
	    matrix.a1, matrix.b1, matrix.c1, matrix.d1,
	    matrix.a2, matrix.b2, matrix.c2, matrix.d2,
	    matrix.a3, matrix.b3, matrix.c3, matrix.d3,
	    matrix.a4, matrix.b4, matrix.c4, matrix.d4);
}

inline glm::mat3 ToMatrix(const aiMatrix3x3 &matrix)
{
	return glm::mat3(
	    matrix.a1, matrix.b1, matrix.c1,
	    matrix.a2, matrix.b2, matrix.c2,
	    matrix.a3, matrix.b3, matrix.c3);
}

inline glm::vec2 ToVector(const aiVector2D &vec)
{
	return glm::vec2(vec.x, vec.y);
}

inline glm::vec3 ToVector(const aiVector3D &vec)
{
	return glm::vec3(vec.x, vec.y, vec.z);
}

inline glm::vec3 ToVector(const aiColor3D &color)
{
	return glm::vec3(color.r, color.g, color.b);
}

inline glm::vec4 ToVector(const aiColor4D &color)
{
	return glm::vec4(color.r, color.g, color.b, color.a);
}

inline glm::quat ToQuaternion(const aiQuaternion &quat)
{
	return glm::quat(quat.w, quat.x, quat.y, quat.z);
}

class AssimpImporter : public Importer<ResourceType::Prefab>
{
  public:
	struct BoneInfo
	{
		uint32_t  id;
		glm::mat4 offset;
	};

	struct ModelInfo
	{
		std::map<size_t, size_t> mesh_table;                // hash - idx
		std::map<size_t, size_t> skinned_mesh_table;        // hash - idx

		std::map<std::string, BoneInfo> bones;

		std::vector<Animation> animations;

		Node root;
	};

	struct MeshletInfo
	{
		std::vector<Meshlet>  meshlets;
		std::vector<uint32_t> meshletdata;
	};

	void ProcessAnimation(RHIContext *rhi_context, ResourceManager *manager, const aiScene *assimp_scene, ModelInfo &data)
	{
		for (uint32_t i = 0; i < assimp_scene->mNumAnimations; i++)
		{
			std::vector<Bone> bones;
			const auto       *assimp_animation = assimp_scene->mAnimations[i];
			for (uint32_t j = 0; j < assimp_animation->mNumChannels; j++)
			{
				auto        channel   = assimp_animation->mChannels[j];
				std::string bone_name = channel->mNodeName.data;
				if (data.bones.find(bone_name) == data.bones.end())
				{
					data.bones[bone_name].id = static_cast<uint32_t>(data.bones.size());
				}

				// Parsing bone key frame data
				uint32_t bone_id = data.bones[bone_name].id;

				std::vector<Bone::KeyPosition> key_positions;
				std::vector<Bone::KeyRotation> key_rotations;
				std::vector<Bone::KeyScale>    key_scales;

				for (uint32_t position_idx = 0; position_idx < channel->mNumPositionKeys; position_idx++)
				{
					Bone::KeyPosition data = {};

					data.position   = ToVector(channel->mPositionKeys[position_idx].mValue);
					data.time_stamp = static_cast<float>(channel->mPositionKeys[position_idx].mTime);
					key_positions.push_back(data);
				}

				for (uint32_t rotation_idx = 0; rotation_idx < channel->mNumRotationKeys; rotation_idx++)
				{
					Bone::KeyRotation data = {};

					data.orientation = ToQuaternion(channel->mRotationKeys[rotation_idx].mValue);
					data.time_stamp  = static_cast<float>(channel->mRotationKeys[rotation_idx].mTime);
					key_rotations.push_back(data);
				}

				for (uint32_t scale_idx = 0; scale_idx < channel->mNumScalingKeys; scale_idx++)
				{
					Bone::KeyScale data = {};

					data.scale      = ToVector(channel->mScalingKeys[scale_idx].mValue);
					data.time_stamp = static_cast<float>(channel->mScalingKeys[scale_idx].mTime);
					key_scales.push_back(data);
				}

				bones.emplace_back(bone_name, bone_id, data.bones.at(bone_name).offset, std::move(key_positions), std::move(key_rotations), std::move(key_scales));
			}

			std::function<void(std::map<std::string, std::pair<glm::mat4, std::string>> &, const Node &, const std::string &, glm::mat4)> build_skeleton =
			    [&](std::map<std::string, std::pair<glm::mat4, std::string>> &hierarchy, const Node &node, std::string parent, glm::mat4 transform) {
				    transform *= node.transform;
				    if (std::find_if(bones.begin(), bones.end(), [&](const Bone &bone) { return node.name == bone.GetBoneName(); }) != bones.end())
				    {
					    hierarchy[node.name] = std::make_pair(transform, parent);
					    parent               = node.name;
				    }
				    for (const auto &child : node.children)
				    {
					    build_skeleton(hierarchy, child, parent, transform);
				    }
			    };

			std::map<std::string, std::pair<glm::mat4, std::string>> hierarchy;
			build_skeleton(hierarchy, data.root, "", glm::mat4(1.f));

			manager->Add<ResourceType::Animation>(rhi_context, assimp_animation->mName.length == 0 ? std::to_string(Hash(assimp_animation)) : assimp_animation->mName.C_Str(), std::move(bones), std::move(hierarchy), static_cast<float>(assimp_animation->mDuration), static_cast<float>(assimp_animation->mTicksPerSecond));
		}
	}

	template <typename T>
	MeshletInfo ProcessMeshlet(std::vector<T> &vertices, std::vector<uint32_t> &indices)
	{
		MeshletInfo meshlet_info;

		const size_t max_vertices  = 64;
		const size_t max_triangles = 124;
		const float  cone_weight   = 0.5f;

		std::vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles));
		std::vector<uint32_t>        meshlet_vertices(meshlets.size() * max_vertices);
		std::vector<uint8_t>         meshlet_triangles(meshlets.size() * max_triangles * 3);

		meshlets.resize(meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), indices.data(), indices.size(), &vertices[0].position.x, vertices.size(), sizeof(T), max_vertices, max_triangles, cone_weight));

		std::unordered_map<size_t, uint32_t> primitive_map;
		for (size_t i = 0; i < indices.size() / 3; i++)
		{
			size_t hash         = Hash(indices[3 * i], indices[3 * i + 1], indices[3 * i + 2]);
			primitive_map[hash] = static_cast<uint32_t>(i);
		}

		for (auto &meshlet : meshlets)
		{
			size_t data_offset = meshlet_info.meshletdata.size();

			for (uint32_t i = 0; i < meshlet.vertex_count; i++)
			{
				meshlet_info.meshletdata.push_back(meshlet_vertices[meshlet.vertex_offset + i]);
			}

			const uint8_t *index_group = &meshlet_triangles[0] + meshlet.triangle_offset;

			for (uint32_t i = 0; i < meshlet.triangle_count; i++)
			{
				size_t hash = Hash((uint32_t) index_group[3 * i] + meshlet.vertex_offset, (uint32_t) index_group[3 * i + 1] + meshlet.vertex_offset, (uint32_t) index_group[3 * i + 2] + meshlet.vertex_offset);
				//if (primitive_map.find(hash) == primitive_map.end())
				//{
				//	LOG_INFO("Fuck");
				//}
				//uint32_t triangle = 0;
				//triangle += index_group[3 * i] & 0xff;
				//triangle += (index_group[3 * i + 1] & 0xff) << 8;
				//triangle += (index_group[3 * i + 2] & 0xff) << 16;
				meshlet_info.meshletdata.push_back(primitive_map.at(hash));
			}

			meshopt_Bounds bounds = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, &vertices[0].position.x, vertices.size(), sizeof(T));

			Meshlet tmp_meshlet = {};

			tmp_meshlet.data_offset    = uint32_t(data_offset);
			tmp_meshlet.triangle_count = meshlet.triangle_count;
			tmp_meshlet.vertex_count   = meshlet.vertex_count;
			tmp_meshlet.vertex_offset   = meshlet.vertex_offset;
			tmp_meshlet.center         = glm::vec3(bounds.center[0], bounds.center[1], bounds.center[2]);
			tmp_meshlet.radius         = bounds.radius;
			tmp_meshlet.cone_axis[0]   = bounds.cone_axis[0];
			tmp_meshlet.cone_axis[1]   = bounds.cone_axis[1];
			tmp_meshlet.cone_axis[2]   = bounds.cone_axis[2];
			tmp_meshlet.cone_cutoff    = bounds.cone_cutoff;

			meshlet_info.meshlets.push_back(tmp_meshlet);
		}

		return meshlet_info;
	}

	std::string ProcessMesh(ResourceManager *manager, RHIContext *rhi_context, const aiScene *assimp_scene, ModelInfo &data, aiMesh *assimp_mesh)
	{
		std::string name = assimp_mesh->mName.C_Str();

		if (manager->Has<ResourceType::Mesh>(name))
		{
			return name;
		}

		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;

		// Parsing vertices
		for (uint32_t j = 0; j < assimp_mesh->mNumVertices; j++)
		{
			Vertex vertex    = {};
			vertex.position  = glm::vec3(assimp_mesh->mVertices[j].x, assimp_mesh->mVertices[j].y, assimp_mesh->mVertices[j].z);
			vertex.normal    = glm::vec3(assimp_mesh->mNormals[j].x, assimp_mesh->mNormals[j].y, assimp_mesh->mNormals[j].z);
			vertex.tangent   = glm::vec3(assimp_mesh->mTangents[j].x, assimp_mesh->mTangents[j].y, assimp_mesh->mTangents[j].z);
			vertex.texcoord0 = assimp_mesh->mTextureCoords[0] ? glm::vec2(assimp_mesh->mTextureCoords[0][j].x, assimp_mesh->mTextureCoords[0][j].y) : glm::vec2(0.f);
			vertex.texcoord1 = assimp_mesh->mTextureCoords[1] ? glm::vec2(assimp_mesh->mTextureCoords[1][j].x, assimp_mesh->mTextureCoords[1][j].y) : glm::vec2(0.f);
			vertices.emplace_back(std::move(vertex));
		}

		// Parsing indices
		for (uint32_t j = 0; j < assimp_mesh->mNumFaces; j++)
		{
			for (uint32_t k = 0; k < 3; k++)
			{
				indices.push_back(assimp_mesh->mFaces[j].mIndices[k]);
			}
		}

		MeshletInfo meshlet_info = ProcessMeshlet(vertices, indices);
		manager->Add<ResourceType::Mesh>(rhi_context, name, std::move(vertices), std::move(indices), std::move(meshlet_info.meshlets), std::move(meshlet_info.meshletdata));
		return name;
	}

	std::string ProcessSkinnedMesh(ResourceManager *manager, RHIContext *rhi_context, const aiScene *assimp_scene, ModelInfo &data, aiMesh *assimp_mesh)
	{
		std::string name = assimp_mesh->mName.C_Str();

		if (manager->Has<ResourceType::SkinnedMesh>(name))
		{
			return name;
		}

		std::vector<SkinnedVertex> vertices;
		std::vector<uint32_t>      indices;

		// Parsing vertices
		for (uint32_t j = 0; j < assimp_mesh->mNumVertices; j++)
		{
			SkinnedVertex vertex = {};
			vertex.position      = glm::vec3(assimp_mesh->mVertices[j].x, assimp_mesh->mVertices[j].y, assimp_mesh->mVertices[j].z);
			vertex.normal        = glm::vec3(assimp_mesh->mNormals[j].x, assimp_mesh->mNormals[j].y, assimp_mesh->mNormals[j].z);
			vertex.tangent       = assimp_mesh->mTangents ? glm::vec3(assimp_mesh->mTangents[j].x, assimp_mesh->mTangents[j].y, assimp_mesh->mTangents[j].z) : glm::vec3(0.f);
			vertex.texcoord0     = assimp_mesh->mTextureCoords[0] ? glm::vec2(assimp_mesh->mTextureCoords[0][j].x, assimp_mesh->mTextureCoords[0][j].y) : glm::vec2(0.f);
			vertex.texcoord1     = assimp_mesh->mTextureCoords[1] ? glm::vec2(assimp_mesh->mTextureCoords[1][j].x, assimp_mesh->mTextureCoords[1][j].y) : glm::vec2(0.f);
			vertices.emplace_back(std::move(vertex));
		}

		// Parsing indices
		for (uint32_t j = 0; j < assimp_mesh->mNumFaces; j++)
		{
			for (uint32_t k = 0; k < 3; k++)
			{
				indices.push_back(assimp_mesh->mFaces[j].mIndices[k]);
			}
		}

		// Parsing bones
		for (uint32_t bone_index = 0; bone_index < assimp_mesh->mNumBones; bone_index++)
		{
			int32_t     bone_id   = -1;
			std::string bone_name = assimp_mesh->mBones[bone_index]->mName.C_Str();

			if (data.bones.find(bone_name) == data.bones.end())
			{
				BoneInfo bone = {};
				bone.id       = static_cast<uint32_t>(data.bones.size());
				bone.offset   = ToMatrix(assimp_mesh->mBones[bone_index]->mOffsetMatrix);

				data.bones[bone_name] = bone;

				bone_id = bone.id;
			}
			else
			{
				bone_id = data.bones.at(bone_name).id;
			}

			data.bones[bone_name] = data.bones.at(bone_name);

			auto     weights    = assimp_mesh->mBones[bone_index]->mWeights;
			uint32_t weight_num = assimp_mesh->mBones[bone_index]->mNumWeights;

			for (uint32_t weight_index = 0; weight_index < weight_num; weight_index++)
			{
				int32_t vertex_id = weights[weight_index].mVertexId;
				float   weight    = weights[weight_index].mWeight;
				for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
				{
					if (vertices[vertex_id].bones[i] < 0)
					{
						vertices[vertex_id].weights[i] = weight;
						vertices[vertex_id].bones[i]   = bone_id;
						break;
					}
				}
			}
		}

		MeshletInfo meshlet_info = ProcessMeshlet(vertices, indices);
		manager->Add<ResourceType::SkinnedMesh>(rhi_context, name, std::move(vertices), std::move(indices), std::move(meshlet_info.meshlets), std::move(meshlet_info.meshletdata));
		return name;
	}

	Node ProcessNode(ResourceManager *manager, RHIContext *rhi_context, const aiScene *assimp_scene, aiNode *assimp_node, ModelInfo &data, aiMatrix4x4 transform = aiMatrix4x4())
	{
		Node node = {};

		node.transform = ToMatrix(assimp_node->mTransformation);
		node.name      = assimp_node->mName.C_Str();

		for (uint32_t i = 0; i < assimp_node->mNumMeshes; i++)
		{
			aiMesh *assimp_mesh = assimp_scene->mMeshes[assimp_node->mMeshes[i]];
			if (assimp_mesh->HasBones())
			{
				node.resources.emplace_back(std::make_pair(ResourceType::SkinnedMesh, ProcessSkinnedMesh(manager, rhi_context, assimp_scene, data, assimp_mesh)));
			}
			else
			{
				node.resources.emplace_back(std::make_pair(ResourceType::Mesh, ProcessMesh(manager, rhi_context, assimp_scene, data, assimp_mesh)));
			}
		}

		for (uint32_t i = 0; i < assimp_node->mNumChildren; i++)
		{
			Node child = ProcessNode(manager, rhi_context, assimp_scene, assimp_node->mChildren[i], data, transform);
			node.children.emplace_back(std::move(child));
		}

		return node;
	}

  protected:
	virtual void Import_(ResourceManager *manager, const std::string &path, RHIContext *rhi_context) override
	{
		if (manager->Has<ResourceType::Prefab>(path))
		{
			return;
		}

		Assimp::Importer importer;

		if (const aiScene *scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace))
		{
			LOG_INFO("Model {} preprocess finish", path);

			ModelInfo   data;
			aiMatrix4x4 identity;
			data.root = ProcessNode(manager, rhi_context, scene, scene->mRootNode, data, identity);
			ProcessAnimation(rhi_context, manager, scene, data);

			manager->Add<ResourceType::Prefab>(path, std::move(data.root));
		}
	}
};

extern "C"
{
	EXPORT_API AssimpImporter *Create()
	{
		return new AssimpImporter;
	}
}