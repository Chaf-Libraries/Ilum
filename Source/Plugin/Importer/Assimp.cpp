#include <Resource/Importer.hpp>
#include <Resource/Resource/Animation.hpp>
#include <Resource/Resource/Mesh.hpp>
#include <Resource/Resource/Prefab.hpp>
#include <Resource/Resource/SkinnedMesh.hpp>
#include <Resource/Resource/Texture2D.hpp>
#include <Resource/ResourceManager.hpp>

#include <Geometry/Meshlet.hpp>

#include <assimp/DefaultLogger.hpp>
#include <assimp/Importer.hpp>
#include <assimp/material.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <meshoptimizer.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <stb_image.h>

using namespace Ilum;

using Vertex        = Resource<ResourceType::Mesh>::Vertex;
using SkinnedVertex = Resource<ResourceType::SkinnedMesh>::SkinnedVertex;
using Node          = Resource<ResourceType::Prefab>::Node;

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
		std::map<std::string, BoneInfo> bones;

		std::vector<std::string> animations;

		Node root;

		std::map<std::string, std::unordered_set<uint32_t>> skinned_mesh_bones;
	};

	struct MeshletInfo
	{
		std::vector<Meshlet>  meshlets;
		std::vector<uint32_t> meshletdata;
	};

	void ProcessAnimation(ResourceManager *manager, RHIContext *rhi_context, const std::string &path, uint32_t animation_id, const aiScene *assimp_scene, ModelInfo &data)
	{
		std::vector<Bone> bones;

		std::string animation_name   = fmt::format("{}.animation.{}", Path::GetInstance().ValidFileName(path), animation_id);
		const auto *assimp_animation = assimp_scene->mAnimations[animation_id];

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
				data.time_stamp = static_cast<float>(channel->mPositionKeys[position_idx].mTime / assimp_animation->mTicksPerSecond);
				key_positions.push_back(data);
			}

			for (uint32_t rotation_idx = 0; rotation_idx < channel->mNumRotationKeys; rotation_idx++)
			{
				Bone::KeyRotation data = {};

				data.orientation = ToQuaternion(channel->mRotationKeys[rotation_idx].mValue);
				data.time_stamp  = static_cast<float>(channel->mRotationKeys[rotation_idx].mTime / assimp_animation->mTicksPerSecond);
				key_rotations.push_back(data);
			}

			for (uint32_t scale_idx = 0; scale_idx < channel->mNumScalingKeys; scale_idx++)
			{
				Bone::KeyScale data = {};

				data.scale      = ToVector(channel->mScalingKeys[scale_idx].mValue);
				data.time_stamp = static_cast<float>(channel->mScalingKeys[scale_idx].mTime / assimp_animation->mTicksPerSecond);
				key_scales.push_back(data);
			}

			bones.emplace_back(bone_name, bone_id, data.bones.at(bone_name).offset, std::move(key_positions), std::move(key_rotations), std::move(key_scales));
		}

		if (bones.size() < data.bones.size())
		{
			for (auto &[name, bone] : data.bones)
			{
				bool found = false;
				for (auto &anim_bone : bones)
				{
					if (anim_bone.GetBoneName() == name)
					{
						found = true;
						break;
					}
				}
				if (!found)
				{
					bones.emplace_back(name, bone.id, data.bones.at(name).offset, std::vector<Bone::KeyPosition>{}, std::vector<Bone::KeyRotation>{}, std::vector<Bone::KeyScale>{});
				}
			}
		}

		std::function<void(HierarchyNode &, Node &)> build_skeleton =
		    [&](HierarchyNode &hierarchy, Node &node) {
			    uint32_t skinned_mesh_count = 0;
			    uint32_t animation_count    = 0;

			    std::vector<std::pair<ResourceType, std::string>> animations;
			    for (auto &[type, uuid] : node.resources)
			    {
				    if (type == ResourceType::SkinnedMesh)
				    {
					    skinned_mesh_count++;
					    if (skinned_mesh_count > animation_count)
					    {
						    for (auto &bone : bones)
						    {
							    if (data.skinned_mesh_bones[uuid].find(bone.GetBoneID()) != data.skinned_mesh_bones[uuid].end())
							    {
								    animations.push_back(std::make_pair(ResourceType::Animation, animation_name));
								    animation_count++;
								    break;
							    }
						    }
					    }
				    }
			    }
			    node.resources.insert(node.resources.end(), animations.begin(), animations.end());

			    hierarchy.name      = node.name;
			    hierarchy.transform = node.transform;
			    for (auto &child : node.children)
			    {
				    HierarchyNode node;
				    build_skeleton(node, child);
				    hierarchy.children.push_back(node);
			    }
		    };

		HierarchyNode hierarchy;
		build_skeleton(hierarchy, data.root);

		data.animations.push_back(animation_name);
		manager->Add<ResourceType::Animation>(rhi_context, animation_name, std::move(bones), std::move(hierarchy));
	}

	template <typename T>
	MeshletInfo ProcessMeshlet(std::vector<T> &vertices, std::vector<uint32_t> &indices)
	{
		MeshletInfo meshlet_info;

		const size_t max_vertices  = 64;
		const size_t max_triangles = 124;
		const float  cone_weight   = 0.5f;

		std::vector<meshopt_Meshlet> meshlets(meshopt_buildMeshletsBound(indices.size(), max_vertices, max_triangles));

		std::vector<uint32_t> meshlet_vertices(meshlets.size() * max_vertices);
		std::vector<uint8_t>  meshlet_triangles(meshlets.size() * max_triangles * 3);

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
				size_t hash = Hash(meshlet_vertices[(uint32_t) index_group[3 * i] + meshlet.vertex_offset], meshlet_vertices[(uint32_t) index_group[3 * i + 1] + meshlet.vertex_offset], meshlet_vertices[(uint32_t) index_group[3 * i + 2] + meshlet.vertex_offset]);
				meshlet_info.meshletdata.push_back(primitive_map.at(hash));
			}

			for (uint32_t i = 0; i < meshlet.triangle_count; i++)
			{
				uint32_t triangle = 0;
				triangle += (uint32_t) index_group[3 * i];
				triangle += (uint32_t) index_group[3 * i + 1] << 8;
				triangle += (uint32_t) index_group[3 * i + 2] << 16;
				meshlet_info.meshletdata.push_back(triangle);
			}

			meshopt_Bounds bounds = meshopt_computeMeshletBounds(&meshlet_vertices[meshlet.vertex_offset], &meshlet_triangles[meshlet.triangle_offset], meshlet.triangle_count, &vertices[0].position.x, vertices.size(), sizeof(T));

			Meshlet tmp_meshlet = {};

			tmp_meshlet.data_offset    = uint32_t(data_offset);
			tmp_meshlet.triangle_count = meshlet.triangle_count;
			tmp_meshlet.vertex_count   = meshlet.vertex_count;
			tmp_meshlet.vertex_offset  = meshlet.vertex_offset;
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

	void ProcessMesh(ResourceManager *manager, RHIContext *rhi_context, const std::string &path, uint32_t mesh_id, const aiScene *assimp_scene, ModelInfo &data)
	{
		std::string mesh_name   = fmt::format("{}.mesh.{}", Path::GetInstance().ValidFileName(path), mesh_id);
		aiMesh     *assimp_mesh = assimp_scene->mMeshes[mesh_id];

		if (manager->Has<ResourceType::Mesh>(mesh_name))
		{
			return;
		}

		std::vector<Vertex>   vertices;
		std::vector<uint32_t> indices;

		// Parsing vertices
		for (uint32_t j = 0; j < assimp_mesh->mNumVertices; j++)
		{
			Vertex vertex    = {};
			vertex.position  = glm::vec3(assimp_mesh->mVertices[j].x, assimp_mesh->mVertices[j].y, assimp_mesh->mVertices[j].z);
			vertex.normal    = assimp_mesh->mNormals ? glm::vec3(assimp_mesh->mNormals[j].x, assimp_mesh->mNormals[j].y, assimp_mesh->mNormals[j].z) : glm::vec3(0.f);
			vertex.tangent   = assimp_mesh->mTangents ? glm::vec3(assimp_mesh->mTangents[j].x, assimp_mesh->mTangents[j].y, assimp_mesh->mTangents[j].z) : glm::vec3(0.f);
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
		manager->Add<ResourceType::Mesh>(rhi_context, mesh_name, std::move(vertices), std::move(indices), std::move(meshlet_info.meshlets), std::move(meshlet_info.meshletdata));
	}

	void ProcessSkinnedMesh(ResourceManager *manager, RHIContext *rhi_context, const std::string &path, uint32_t mesh_id, const aiScene *assimp_scene, ModelInfo &data)
	{
		std::string mesh_name   = fmt::format("{}.mesh.{}", Path::GetInstance().ValidFileName(path), mesh_id);
		aiMesh     *assimp_mesh = assimp_scene->mMeshes[mesh_id];

		if (manager->Has<ResourceType::SkinnedMesh>(mesh_name))
		{
			return;
		}

		std::vector<SkinnedVertex>   vertices;
		std::vector<uint32_t>        indices;
		std::unordered_set<uint32_t> bones;

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

			bones.insert(bone_id);

			data.bones[bone_name] = data.bones.at(bone_name);

			auto     weights    = assimp_mesh->mBones[bone_index]->mWeights;
			uint32_t weight_num = assimp_mesh->mBones[bone_index]->mNumWeights;

			for (uint32_t weight_index = 0; weight_index < weight_num; weight_index++)
			{
				int32_t vertex_id = weights[weight_index].mVertexId;
				float   weight    = weights[weight_index].mWeight;

				for (int i = 0; i < MAX_BONE_INFLUENCE; ++i)
				{
					if (vertices[vertex_id].bones[i] < 0 && weight > 0.f)
					{
						vertices[vertex_id].weights[i] = weight;
						vertices[vertex_id].bones[i]   = bone_id;
						break;
					}
				}
			}
		}

		MeshletInfo meshlet_info = ProcessMeshlet(vertices, indices);

		data.skinned_mesh_bones[mesh_name] = bones;

		manager->Add<ResourceType::SkinnedMesh>(rhi_context, mesh_name, std::move(vertices), std::move(indices), std::move(meshlet_info.meshlets), std::move(meshlet_info.meshletdata));
	}

	Node ProcessNode(ResourceManager *manager, RHIContext *rhi_context, const std::string &path, const aiScene *assimp_scene, aiNode *assimp_node, ModelInfo &data, aiMatrix4x4 transform = aiMatrix4x4())
	{
		Node node = {};

		node.transform = ToMatrix(assimp_node->mTransformation);
		node.name      = assimp_node->mName.C_Str();

		for (uint32_t i = 0; i < assimp_node->mNumMeshes; i++)
		{
			aiMesh     *assimp_mesh = assimp_scene->mMeshes[assimp_node->mMeshes[i]];
			std::string mesh_name   = fmt::format("{}.mesh.{}", Path::GetInstance().ValidFileName(path), assimp_node->mMeshes[i]);

			if (assimp_mesh->HasBones())
			{
				node.resources.emplace_back(std::make_pair(ResourceType::SkinnedMesh, mesh_name));
			}
			else
			{
				node.resources.emplace_back(std::make_pair(ResourceType::Mesh, mesh_name));
			}
		}

		for (uint32_t i = 0; i < assimp_node->mNumChildren; i++)
		{
			Node child = ProcessNode(manager, rhi_context, path, assimp_scene, assimp_node->mChildren[i], data, transform);
			node.children.emplace_back(std::move(child));
		}

		return node;
	}

	void ProcessMaterial(ResourceManager *manager, RHIContext *rhi_context, const std::string &path, uint32_t material_id, const aiScene *assimp_scene)
	{
		std::string material_name   = fmt::format("{}.material.{}", Path::GetInstance().ValidFileName(path), material_id);
		aiMaterial *assimp_material = assimp_scene->mMaterials[material_id];

		float roughness          = 0.f;
		float emissive_intensity = 0.f;

		// Base Color
		{
			glm::vec4 base_color = glm::vec4(0.f);
			aiString  color_texture;

			assimp_material->Get(AI_MATKEY_BASE_COLOR, base_color.x);
			assimp_material->GetTexture(AI_MATKEY_BASE_COLOR_TEXTURE, &color_texture);
			ProcessTexture(manager, rhi_context, path, assimp_scene, color_texture.C_Str());
		}

		// Metallic
		{
			float    metallic = 0.f;
			aiString metallic_texture;

			assimp_material->Get(AI_MATKEY_METALLIC_FACTOR, metallic);
			assimp_material->GetTexture(AI_MATKEY_METALLIC_TEXTURE, &metallic_texture);
			ProcessTexture(manager, rhi_context, path, assimp_scene, metallic_texture.C_Str());
		}

		// Roughness
		{
			float    roughness = 0.f;
			aiString roughness_texture;

			assimp_material->Get(AI_MATKEY_ROUGHNESS_FACTOR, roughness);
			assimp_material->GetTexture(AI_MATKEY_ROUGHNESS_TEXTURE, &roughness_texture);
			ProcessTexture(manager, rhi_context, path, assimp_scene, roughness_texture.C_Str());
		}

		assimp_material->Get(AI_MATKEY_EMISSIVE_INTENSITY, emissive_intensity);
	}

	void ProcessTexture(ResourceManager *manager, RHIContext *rhi_context, const std::string &path, const aiScene *assimp_scene, const std::string &filename)
	{
		auto [assimp_texture, texture_id] = assimp_scene->GetEmbeddedTextureAndIndex(filename.c_str());
		if (texture_id < 0 && filename.empty())
		{
			return;
		}

		if (texture_id < 0)
		{
			// External texture
			auto &importer = Importer<ResourceType::Texture2D>::GetInstance("STB");
			if (importer)
			{
				importer->Import(manager, Path::GetInstance().GetFileDirectory(path) + filename, rhi_context);
			}
		}
		else
		{
			std::string texture_name = fmt::format("{}.texture.{}", Path::GetInstance().ValidFileName(path), texture_id);
			if (manager->Has<ResourceType::Texture2D>(texture_name))
			{
				return;
			}

			TextureDesc desc = {};
			desc.name        = texture_name;
			desc.width       = 1;
			desc.height      = 1;
			desc.depth       = 1;
			desc.mips        = 1;
			desc.layers      = 1;
			desc.samples     = 1;

			void   *raw_data = nullptr;
			size_t  size     = 0;
			int32_t width = 0, height = 0, channel = 0;

			const int32_t req_channel = 4;

			stbi_uc *assimp_texture_data     = reinterpret_cast<stbi_uc *>(assimp_texture->pcData);
			int32_t  assimp_texture_data_len = static_cast<int32_t>(assimp_texture->mWidth * glm::max(1u, assimp_texture->mHeight) * 4);

			if (stbi_is_hdr_from_memory(assimp_texture_data, assimp_texture_data_len))
			{
				raw_data    = stbi_loadf_from_memory(assimp_texture_data, assimp_texture_data_len, &width, &height, &channel, req_channel);
				size        = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(float);
				desc.format = RHIFormat::R32G32B32A32_FLOAT;
			}
			else if (stbi_is_16_bit_from_memory(assimp_texture_data, assimp_texture_data_len))
			{
				raw_data    = stbi_load_16_from_memory(assimp_texture_data, assimp_texture_data_len, &width, &height, &channel, req_channel);
				size        = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint16_t);
				desc.format = RHIFormat::R16G16B16A16_FLOAT;
			}
			else
			{
				raw_data    = stbi_load_from_memory(assimp_texture_data, assimp_texture_data_len, &width, &height, &channel, req_channel);
				size        = static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(req_channel) * sizeof(uint8_t);
				desc.format = RHIFormat::R8G8B8A8_UNORM;
			}

			desc.width  = static_cast<uint32_t>(width);
			desc.height = static_cast<uint32_t>(height);
			desc.mips   = static_cast<uint32_t>(std::floor(std::log2(std::max(width, height))) + 1);
			desc.usage  = RHITextureUsage::ShaderResource | RHITextureUsage::Transfer;

			std::vector<uint8_t> data;

			data.resize(size);
			std::memcpy(data.data(), raw_data, size);

			stbi_image_free(raw_data);

			manager->Add<ResourceType::Texture2D>(rhi_context, std::move(data), desc);
		}
	}

  protected:
	virtual void Import_(ResourceManager *manager, const std::string &path, RHIContext *rhi_context) override
	{
		std::string prefab_name = Path::GetInstance().ValidFileName(path);

		Assimp::Importer importer;

		if (const aiScene *assimp_scene = importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs | aiProcess_GenSmoothNormals | aiProcess_CalcTangentSpace))
		{
			ModelInfo data;

			for (uint32_t i = 0; i < assimp_scene->mNumMeshes; i++)
			{
				aiMesh *assimp_mesh = assimp_scene->mMeshes[i];
				if (assimp_mesh->HasBones())
				{
					ProcessSkinnedMesh(manager, rhi_context, path, i, assimp_scene, data);
				}
				else
				{
					ProcessMesh(manager, rhi_context, path, i, assimp_scene, data);
				}
			}

			for (uint32_t i = 0; i < assimp_scene->mNumMaterials; i++)
			{
				ProcessMaterial(manager, rhi_context, path, i, assimp_scene);
			}

			aiMatrix4x4 identity;
			data.root = ProcessNode(manager, rhi_context, path, assimp_scene, assimp_scene->mRootNode, data, identity);

			for (uint32_t i = 0; i < assimp_scene->mNumAnimations; i++)
			{
				ProcessAnimation(manager, rhi_context, path, i, assimp_scene, data);
			}

			if (!manager->Has<ResourceType::Prefab>(prefab_name))
			{
				manager->Add<ResourceType::Prefab>(rhi_context, prefab_name, std::move(data.root));
			}
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