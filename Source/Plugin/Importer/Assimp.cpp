#include <Resource/Importer.hpp>
#include <Resource/Resource/Animation.hpp>
#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/Prefab.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/ResourceManager.hpp>

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
using BoneInfo      = Resource<ResourceType::Animation>::BoneInfo;

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
	struct ModelInfo
	{
		std::map<size_t, size_t> mesh_table;                // hash - idx
		std::map<size_t, size_t> skinned_mesh_table;        // hash - idx

		std::map<std::string, BoneInfo> bones;

		std::vector<Animation> animations;

		Node root;
	};

	// struct ModelData
	//{
	//	std::vector<Mesh> meshes;

	//	std::map<std::string, Bone> bone_map;
	//};

	// void ProcessMesh(const aiScene *scene, aiMesh *mesh, aiMatrix4x4 transform, ModelData &data)
	//{
	//	Mesh result;

	//	result.name = mesh->mName.C_Str();
	//	result.transform = ToMatrix(transform);

	//	result.vertices.reserve(mesh->mNumVertices);
	//	result.indices.reserve(mesh->mNumFaces * 3U);

	//	// Process vertices
	//	for (uint32_t j = 0; j < mesh->mNumVertices; j++)
	//	{
	//		Vertex vertex;
	//		vertex.position  = glm::vec3(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
	//		vertex.normal    = glm::vec3(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);
	//		vertex.texcoord0 = mesh->mTextureCoords[0] ? glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y) : glm::vec2(0.f);
	//		vertex.texcoord1 = mesh->mTextureCoords[1] ? glm::vec2(mesh->mTextureCoords[1][j].x, mesh->mTextureCoords[1][j].y) : glm::vec2(0.f);
	//		result.vertices.emplace_back(vertex);
	//	}

	//	// Process indices
	//	for (uint32_t j = 0; j < mesh->mNumFaces; j++)
	//	{
	//		for (uint32_t k = 0; k < 3; k++)
	//		{
	//			result.indices.push_back(mesh->mFaces[j].mIndices[k]);
	//		}
	//	}

	//	// Process bones
	//	{
	//		if (mesh->mNumBones > 0)
	//		{
	//			for (uint32_t bone_index = 0; bone_index < mesh->mNumBones; bone_index++)
	//			{
	//				int32_t     bone_id   = -1;
	//				std::string bone_name = mesh->mBones[bone_index]->mName.C_Str();

	//				if (data.bone_map.find(bone_name) == data.bone_map.end())
	//				{
	//					Bone bone   = {};
	//					bone.id     = static_cast<uint32_t>(data.bone_map.size());
	//					bone.offset = ToMatrix(mesh->mBones[bone_index]->mOffsetMatrix);

	//					data.bone_map[bone_name] = bone;

	//					bone_id = bone.id;
	//				}
	//				else
	//				{
	//					bone_id = data.bone_map.at(bone_name).id;
	//				}

	//				result.bones[bone_name] = data.bone_map.at(bone_name);

	//				auto     weights    = mesh->mBones[bone_index]->mWeights;
	//				uint32_t weight_num = mesh->mBones[bone_index]->mNumWeights;

	//				for (uint32_t weight_index = 0; weight_index < weight_num; weight_index++)
	//				{
	//					int32_t vertex_id = weights[weight_index].mVertexId;
	//					float   weight    = weights[weight_index].mWeight;
	//					for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
	//					{
	//						if (result.vertices[vertex_id].bones[i] < 0)
	//						{
	//							result.vertices[vertex_id].weights[i] = weight;
	//							result.vertices[vertex_id].bones[i]   = bone_id;
	//							break;
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}

	//	const size_t max_vertices  = 64;
	//	const size_t max_triangles = 124;
	//	const float  cone_weight   = 0.5f;

	//	// Optimize mesh
	//	meshopt_optimizeVertexCache(result.indices.data(), result.indices.data(), result.indices.size(), result.vertices.size());
	//	meshopt_optimizeOverdraw(result.indices.data(), result.indices.data(), result.indices.size(), &result.vertices[0].position.x, result.vertices.size(), sizeof(Vertex), 1.05f);
	//	meshopt_optimizeVertexFetch(result.vertices.data(), result.indices.data(), result.indices.size(), result.vertices.data(), result.vertices.size(), sizeof(Vertex));

	//	// Generate meshlets
	//	size_t                       max_meshlets = meshopt_buildMeshletsBound(result.indices.size(), max_vertices, max_triangles);
	//	std::vector<meshopt_Meshlet> meshlets(max_meshlets);
	//	std::vector<uint32_t>        meshlet_vertices(max_meshlets * max_vertices);
	//	std::vector<uint8_t>         meshlet_triangles(max_meshlets * max_triangles * 3);
	//	size_t                       meshlet_count = meshopt_buildMeshlets(meshlets.data(), meshlet_vertices.data(), meshlet_triangles.data(), result.indices.data(),
	//	                                                                   result.indices.size(), &result.vertices[0].position.x, result.vertices.size(), sizeof(Vertex), max_vertices, max_triangles, cone_weight);

	//	// Merge meshlets
	//	const meshopt_Meshlet &last = meshlets[meshlet_count - 1];
	//	meshlet_vertices.resize((size_t) last.vertex_offset + last.vertex_count);
	//	meshlet_triangles.resize((size_t) last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
	//	meshlets.resize(meshlet_count);

	//	result.meshlets.reserve(meshlets.size());
	//	for (auto &meshlet : meshlets)
	//	{
	//		auto bound = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset],
	//		                                          meshlet.triangle_count, &result.vertices[0].position.x, result.vertices.size(), sizeof(Vertex));

	//		Meshlet tmp = {};

	//		tmp.meshlet_primitive_offset = meshlet.triangle_offset / 3U;
	//		tmp.meshlet_vertex_offset    = meshlet.vertex_offset;
	//		tmp.vertex_count             = meshlet.vertex_count;
	//		tmp.primitive_count          = meshlet.triangle_count;
	//		tmp.cone_cutoff              = bound.cone_cutoff;
	//		tmp.radius                   = bound.radius;

	//		std::memcpy(&tmp.center, bound.center, sizeof(float) * 3);
	//		std::memcpy(&tmp.cone_axis, bound.cone_axis, sizeof(float) * 3);

	//		result.meshlets.emplace_back(std::move(tmp));
	//	}

	//	result.meshlet_vertices = std::move(meshlet_vertices);
	//	result.meshlet_primitives.reserve(meshlet_triangles.size() / 3);
	//	for (size_t i = 0; i < meshlet_triangles.size() / 3; i++)
	//	{
	//		result.meshlet_primitives.push_back(PackTriangle(meshlet_triangles[3U * i], meshlet_triangles[3U * i + 1U], meshlet_triangles[3U * i + 2U]));
	//	}

	//	data.meshes.emplace_back(std::move(result));
	//}

	// void ProcessNode(const aiScene *scene, aiNode *node, ModelData &data, aiMatrix4x4 transform = aiMatrix4x4())
	//{
	//	transform = transform * node->mTransformation;

	//	for (uint32_t i = 0; i < node->mNumMeshes; i++)
	//	{
	//		aiMesh *mesh = scene->mMeshes[node->mMeshes[i]];
	//		ProcessMesh(scene, mesh, transform, data);
	//	}

	//	for (uint32_t i = 0; i < node->mNumChildren; i++)
	//	{
	//		ProcessNode(scene, node->mChildren[i], data, transform);
	//	}
	//}

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

	// size_t ProcessMesh(const aiScene *scene, aiNode *node, ModelInfo &data, bool &skinned)
	//{
	//	if (node->mNumMeshes == 0)
	//	{
	//		return ~0U;
	//	}

	//	size_t hash = 0;
	//	skinned     = false;
	//	for (uint32_t i = 0; i < node->mNumMeshes; i++)
	//	{
	//		HashCombine(hash, node->mMeshes[i]);
	//		skinned |= scene->mMeshes[node->mMeshes[i]]->HasBones();
	//	}

	//	if (!skinned)
	//	{
	//		if (data.mesh_table.find(hash) != data.mesh_table.end())
	//		{
	//			return data.mesh_table.at(hash);
	//		}

	//		data.mesh_table.emplace(hash, data.meshes.size());

	//		Mesh mesh;

	//		for (uint32_t i = 0; i < node->mNumMeshes; i++)
	//		{
	//			auto *assimp_mesh = scene->mMeshes[node->mMeshes[i]];

	//			Submesh submesh = {};

	//			std::vector<Vertex>   vertices;
	//			std::vector<uint32_t> indices;

	//			submesh.index_offset  = static_cast<uint32_t>(mesh.indices.size());
	//			submesh.vertex_offset = static_cast<uint32_t>(mesh.vertices.size());
	//			submesh.index_count   = assimp_mesh->mNumFaces * 3;
	//			submesh.vertex_count  = assimp_mesh->mNumVertices;

	//			for (uint32_t j = 0; j < assimp_mesh->mNumVertices; j++)
	//			{
	//				Vertex vertex    = {};
	//				vertex.position  = glm::vec3(assimp_mesh->mVertices[j].x, assimp_mesh->mVertices[j].y, assimp_mesh->mVertices[j].z);
	//				vertex.normal    = glm::vec3(assimp_mesh->mNormals[j].x, assimp_mesh->mNormals[j].y, assimp_mesh->mNormals[j].z);
	//				vertex.tangent   = glm::vec3(assimp_mesh->mTangents[j].x, assimp_mesh->mTangents[j].y, assimp_mesh->mTangents[j].z);
	//				vertex.texcoord0 = assimp_mesh->mTextureCoords[0] ? glm::vec2(assimp_mesh->mTextureCoords[0][j].x, assimp_mesh->mTextureCoords[0][j].y) : glm::vec2(0.f);
	//				vertex.texcoord1 = assimp_mesh->mTextureCoords[1] ? glm::vec2(assimp_mesh->mTextureCoords[1][j].x, assimp_mesh->mTextureCoords[1][j].y) : glm::vec2(0.f);
	//				vertices.emplace_back(std::move(vertex));
	//			}

	//			for (uint32_t j = 0; j < assimp_mesh->mNumFaces; j++)
	//			{
	//				for (uint32_t k = 0; k < 3; k++)
	//				{
	//					indices.push_back(assimp_mesh->mFaces[j].mIndices[k]);
	//				}
	//			}

	//			mesh.vertices.insert(mesh.vertices.end(), std::move_iterator(vertices.begin()), std::move_iterator(vertices.end()));
	//			mesh.indices.insert(mesh.indices.end(), std::move_iterator(indices.begin()), std::move_iterator(indices.end()));
	//			mesh.submeshes.emplace_back(std::move(submesh));
	//		}

	//		data.meshes.emplace_back(std::move(mesh));
	//		return data.mesh_table.at(hash);
	//	}
	//	else
	//	{
	//		if (data.skinned_mesh_table.find(hash) != data.skinned_mesh_table.end())
	//		{
	//			return data.skinned_mesh_table.at(hash);
	//		}

	//		data.skinned_mesh_table.emplace(hash, data.meshes.size());

	//		/*SkinnedMesh skinned_mesh = {};

	//		for (uint32_t i = 0; i < node->mNumMeshes; i++)
	//		{
	//			auto *assimp_mesh = scene->mMeshes[node->mMeshes[i]];

	//			Submesh submesh = {};

	//			std::vector<SkinnedVertex> vertices;
	//			std::vector<uint32_t>      indices;

	//			submesh.index_offset  = static_cast<uint32_t>(skinned_mesh.indices.size());
	//			submesh.vertex_offset = static_cast<uint32_t>(skinned_mesh.vertices.size());
	//			submesh.index_count   = assimp_mesh->mNumFaces * 3;
	//			submesh.vertex_count  = assimp_mesh->mNumVertices;

	//			for (uint32_t j = 0; j < assimp_mesh->mNumVertices; j++)
	//			{
	//				SkinnedVertex vertex = {};
	//				vertex.position      = glm::vec3(assimp_mesh->mVertices[j].x, assimp_mesh->mVertices[j].y, assimp_mesh->mVertices[j].z);
	//				vertex.normal        = glm::vec3(assimp_mesh->mNormals[j].x, assimp_mesh->mNormals[j].y, assimp_mesh->mNormals[j].z);
	//				vertex.tangent       = glm::vec3(assimp_mesh->mTangents[j].x, assimp_mesh->mTangents[j].y, assimp_mesh->mTangents[j].z);
	//				vertex.texcoord0     = assimp_mesh->mTextureCoords[0] ? glm::vec2(assimp_mesh->mTextureCoords[0][j].x, assimp_mesh->mTextureCoords[0][j].y) : glm::vec2(0.f);
	//				vertex.texcoord1     = assimp_mesh->mTextureCoords[1] ? glm::vec2(assimp_mesh->mTextureCoords[1][j].x, assimp_mesh->mTextureCoords[1][j].y) : glm::vec2(0.f);
	//				vertices.emplace_back(std::move(vertex));
	//			}

	//			for (uint32_t j = 0; j < assimp_mesh->mNumFaces; j++)
	//			{
	//				for (uint32_t k = 0; k < 3; k++)
	//				{
	//					indices.push_back(assimp_mesh->mFaces[j].mIndices[k]);
	//				}
	//			}

	//			for (uint32_t bone_index = 0; bone_index < assimp_mesh->mNumBones; bone_index++)
	//			{
	//				int32_t     bone_id   = -1;
	//				std::string bone_name = assimp_mesh->mBones[bone_index]->mName.C_Str();

	//				if (data.bones.find(bone_name) == data.bones.end())
	//				{
	//					BoneInfo bone = {};
	//					bone.id       = static_cast<uint32_t>(data.bones.size());
	//					bone.offset   = ToMatrix(assimp_mesh->mBones[bone_index]->mOffsetMatrix);

	//					data.bones[bone_name] = bone;

	//					bone_id = bone.id;
	//				}
	//				else
	//				{
	//					bone_id = data.bones.at(bone_name).id;
	//				}

	//				data.bones[bone_name] = data.bones.at(bone_name);

	//				auto     weights    = assimp_mesh->mBones[bone_index]->mWeights;
	//				uint32_t weight_num = assimp_mesh->mBones[bone_index]->mNumWeights;

	//				for (uint32_t weight_index = 0; weight_index < weight_num; weight_index++)
	//				{
	//					int32_t vertex_id = weights[weight_index].mVertexId;
	//					float   weight    = weights[weight_index].mWeight;
	//					for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
	//					{
	//						if (vertices[vertex_id].bones[i] < 0)
	//						{
	//							vertices[vertex_id].weights[i] = weight;
	//							vertices[vertex_id].bones[i]   = bone_id;
	//							break;
	//						}
	//					}
	//				}
	//			}

	//			skinned_mesh.vertices.insert(skinned_mesh.vertices.end(), std::move_iterator(vertices.begin()), std::move_iterator(vertices.end()));
	//			skinned_mesh.indices.insert(skinned_mesh.indices.end(), std::move_iterator(indices.begin()), std::move_iterator(indices.end()));
	//			skinned_mesh.submeshes.emplace_back(std::move(submesh));
	//		}

	//		data.skinned_meshes.emplace_back(std::move(skinned_mesh));*/
	//		return data.skinned_mesh_table.at(hash);
	//	}

	//	return ~0U;
	//}

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

		manager->Add<ResourceType::Mesh>(rhi_context, name, std::move(vertices), std::move(indices));
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

		manager->Add<ResourceType::SkinnedMesh>(rhi_context, name, std::move(vertices), std::move(indices));
		return name;
	}

	Node ProcessNode(ResourceManager *manager, RHIContext *rhi_context, const aiScene *assimp_scene, aiNode *assimp_node, ModelInfo &data, aiMatrix4x4 transform = aiMatrix4x4())
	{
		Node node = {};

		//transform = transform * assimp_node->mTransformation;

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

			std::function<void(const Node &, glm::mat4)> func = [&](const Node &node, glm::mat4 parent) {
				if (data.bones.find(node.name) != data.bones.end())
				{
					glm::mat4 offset             = data.bones[node.name].offset;
					parent = parent * node.transform*offset;
					data.bones[node.name].offset = parent;
				}
				for (auto &child : node.children)
				{
					func(child, parent);
				}
			};

			//func(data.root, glm::mat4(1.f));

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