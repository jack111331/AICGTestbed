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

#define NoTextureRootSignature \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | " \
              "DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
              "CBV(b0)," \
              "DescriptorTable ( SRV(t0), visibility = SHADER_VISIBILITY_PIXEL ),"\
              "DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL )"

Texture2D<float4> BaseColorTexture : register(t0);
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