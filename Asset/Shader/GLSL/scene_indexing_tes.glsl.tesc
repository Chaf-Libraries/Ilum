#version 450 core      

layout (vertices = 3) out;

// Input
layout (location = 0) in vec3 inPos[];
layout (location = 1) in vec3 inNormal[];
layout (location = 2) in vec3 inColor[];
layout (location = 3) in vec2 inUV[];
layout (location = 4) in vec4 inTangent[];
layout (location = 5) in mat3 inTBN[];
layout (location = 8) flat in uint inIndex[];

// Output
struct PosPatch
{
    vec3 pos[9];
};

layout (location = 0) out patch PosPatch outPosPatch;

// layout (location = 0) out vec3 outPosPatch[3][9];
layout (location = 9) out vec3 outNormal[3];
layout (location = 10) out vec3 outColor[3];
layout (location = 11) out vec2 outUV[3];
layout (location = 12) out vec4 outTangent[3];
layout (location = 13) out mat3 outTBN[3];
layout (location = 16) out uint outIndex[3];

// Uniform buffer
layout (set = 0, binding = 0) uniform UBOScene 
{
	mat4 projection;
	mat4 view;
	vec4 lightPos;
	vec4 viewPos;
    vec4 frustum[6];
	vec4 range;
} uboScene;

vec3 ProjectToPlane(vec3 Point, vec3 PlanePoint, vec3 PlaneNormal)                              
{                                                                                              
    vec3 v = Point - PlanePoint;                                                                
    float Len = dot(v, PlaneNormal);                                                           
    vec3 d = Len * PlaneNormal;                                                                 
    return (Point - d);                                                                        
}                                                                                               


float ComputeTessLevel(vec3 p0, vec3 p1)
{
    mat4 mvp = uboScene.projection* uboScene.view;
    vec4 pa = mvp * vec4(p0, 1.0);
    vec4 pb = mvp * vec4(p1, 1.0);
	vec3 d=pa.xyz/pa.w-pb.xyz/pb.w;
    float diameter = sqrt(d.x*d.x+d.y*d.y);
	
	//这里的20控制细分层次,可以根据屏幕大小等调整
   float tessLevel = max(1.0,   ceil(uboScene.range.x*diameter/10));
   
   return min(tessLevel, 20);
}


void main()
{
    vec3 pa = inPos[0];
    vec3 pb = inPos[1];
    vec3 pc = inPos[2];

    vec3 na = inTBN[0][2];
    vec3 nb = inTBN[1][2];
    vec3 nc = inTBN[2][2];

    float tessa = ComputeTessLevel(pa, pb);
    float tessb = ComputeTessLevel(pb, pc);
    float tessc = ComputeTessLevel(pc, pa);


    gl_TessLevelOuter[0] = tessb;
    gl_TessLevelOuter[1] = tessc;
    gl_TessLevelOuter[2] = tessa;
    gl_TessLevelInner[0] = max(tessa, max(tessb, tessc));

    // Output position patch 1
    
    outPosPatch.pos[0] = pa;
    outPosPatch.pos[1] = ProjectToPlane(pa, pb, nb);
    outPosPatch.pos[2] = ProjectToPlane(pa, pc, nc);

    outPosPatch.pos[3] = ProjectToPlane(pb, pa, na);
    outPosPatch.pos[4] = pb;
    outPosPatch.pos[5] = ProjectToPlane(pb, pc, nc);
    
    outPosPatch.pos[6] = ProjectToPlane(pc, pa, na);
    outPosPatch.pos[7] = ProjectToPlane(pc, pb, nb);
    outPosPatch.pos[8] = pc;  

    outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    outColor[gl_InvocationID] = inColor[gl_InvocationID];
    outUV[gl_InvocationID] = inUV[gl_InvocationID];
    outTangent[gl_InvocationID] = inTangent[gl_InvocationID];
    outTBN[gl_InvocationID] = inTBN[gl_InvocationID];
    outIndex[gl_InvocationID] = inIndex[gl_InvocationID];
}






