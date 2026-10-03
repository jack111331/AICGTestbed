struct VSInput
{
    float3 Position : Position;
    float2 TexCoord: TexCoord0;
};

struct VSOutput
{
    float4 Position : SV_Position;
    float2 TexCoord : TexCoord0;
};

struct PSInput
{
    float4 Diffuse  : COLOR0;
    float4 Specular : COLOR1;
};

cbuffer PBR_Constants : register(b0)
{
    float3   PBR_EyePosition            : packoffset(c0);
    float4x4 PBR_World                  : packoffset(c1);
    float3x3 PBR_WorldInverseTranspose  : packoffset(c5);
    float4x4 PBR_WorldViewProj          : packoffset(c8);
    float4x4 PBR_PrevWorldViewProj      : packoffset(c12);

    float3 PBR_LightDirection[3]        : packoffset(c16);
    float3 PBR_LightColor[3]            : packoffset(c19);   // "Specular and diffuse light" in PBR
 
    float3 PBR_ConstantAlbedo           : packoffset(c22);   // Constant values if not a textured effect
    float  PBR_ConstantMetallic         : packoffset(c23.x);
    float  PBR_ConstantRoughness        : packoffset(c23.y);

    int PBR_NumRadianceMipLevels        : packoffset(c23.z);

    // Size of render target
    float PBR_TargetWidth               : packoffset(c23.w);
    float PBR_TargetHeight              : packoffset(c24.x);
};

// Per-primitive material parameters. Mirrors MaterialConstants in
// Engine/SceneGraph.hpp -- the packoffsets below and that struct must be
// changed together.
//
// Nothing here is consumed yet: the factors and flags are uploaded and bound
// so the PBR shading can be written against them.
cbuffer MaterialConstants : register(b1)
{
    float4 Mat_BaseColorFactor              : packoffset(c0);

    float3 Mat_EmissiveFactor               : packoffset(c1);
    float  Mat_EmissiveStrength             : packoffset(c1.w);

    float  Mat_MetallicFactor               : packoffset(c2.x);
    float  Mat_RoughnessFactor              : packoffset(c2.y);
    float  Mat_NormalScale                  : packoffset(c2.z);
    float  Mat_OcclusionStrength            : packoffset(c2.w);

    float  Mat_AlphaCutoff                  : packoffset(c3.x);
    int    Mat_AlphaMode                    : packoffset(c3.y);   // 0 opaque, 1 mask, 2 blend
    int    Mat_DoubleSided                  : packoffset(c3.z);
    float  Mat_Ior                          : packoffset(c3.w);

    // Whether each slot has a real texture. Absent slots are still bound, to
    // a 1x1 white fallback, so sampling them is safe either way.
    int    Mat_HasBaseColorTexture          : packoffset(c4.x);
    int    Mat_HasMetallicRoughnessTexture  : packoffset(c4.y);
    int    Mat_HasNormalTexture             : packoffset(c4.z);
    int    Mat_HasOcclusionTexture          : packoffset(c4.w);
    int    Mat_HasEmissiveTexture           : packoffset(c5.x);

    // TEXCOORD set each texture samples. Only TEXCOORD_0 is uploaded so far.
    int    Mat_BaseColorTexCoord            : packoffset(c5.y);
    int    Mat_MetallicRoughnessTexCoord    : packoffset(c5.z);
    int    Mat_NormalTexCoord               : packoffset(c5.w);
    int    Mat_OcclusionTexCoord            : packoffset(c6.x);
    int    Mat_EmissiveTexCoord             : packoffset(c6.y);
};

#define NoTextureRootSignature \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | " \
              "DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
              "CBV(b0), " \
              "DescriptorTable ( SRV(t0, numDescriptors = 5), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "CBV(b1)"

// The five glTF metallic-roughness texture slots, in the order the SRV
// descriptor table stages them. Slots whose material has no texture receive
// the white fallback, so every one is always safe to sample.
Texture2D<float4> BaseColorTexture         : register(t0);
Texture2D<float4> MetallicRoughnessTexture : register(t1);
Texture2D<float4> NormalTexture            : register(t2);
Texture2D<float4> OcclusionTexture         : register(t3);
Texture2D<float4> EmissiveTexture          : register(t4);
sampler Sampler : register(s0);

// Vertex shader: self-created quad.
[RootSignature(NoTextureRootSignature)]
VSOutput VSStraight(VSInput v)
{
    VSOutput vout;

    vout.Position = mul(float4(v.Position, 1.0), PBR_WorldViewProj);
    vout.TexCoord = v.TexCoord;
    // vout.Position = float4(v.Position, 1.0);
    return vout;
}


//--------------------------------------------------------------------------------------
// Pixel shader: pass-through
[RootSignature(NoTextureRootSignature)]
float4 PSStraight(VSOutput pin) : SV_Target0
{
    return BaseColorTexture.Sample(Sampler, pin.TexCoord);
}