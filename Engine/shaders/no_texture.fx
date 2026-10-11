struct VSInput
{
    float3 Position : Position;
    float2 TexCoord: TexCoord0;
    float3 Normal   : Normal;
    // glTF TANGENT is VEC4: xyz is the tangent, w is +1/-1 giving the handedness
    // of the bitangent, which is cross(normal, tangent) * w. Reads as 0 on a
    // primitive that supplied no tangents -- check Mat_HasTangents, not length.
    float4 Tangent  : Tangent;

    // Skinning. JOINTS_0 indexes SkinJoints below; WEIGHTS_0 are the blend
    // weights, which glTF says should sum to 1 but does not guarantee. Both
    // read as 0 on an unskinned primitive -- branch on Mat_IsSkinned rather
    // than inspecting them.
    uint4  Joints   : Joints0;
    float4 Weights  : Weights0;
};

struct VSOutput
{
    float4 NDCPosition : SV_Position;
    float3 WorldPosition : Position;
    float2 TexCoord : TexCoord0;
    float3 WorldNormal : Normal;
    // xyz in world space, w carried through from the attribute.
    float4 WorldTangent : Tangent;
};

struct PSInput
{
    float4 Diffuse  : COLOR0;
    float4 Specular : COLOR1;
};

static const float M_PI = 3.14159265359;

// One glTF KHR_lights_punctual light, already resolved into world space by
// SceneGraph: the node transform that placed and aimed it has been applied on
// the CPU, so nothing here needs transforming.
//
// Mirrors struct ShaderLight in SceneGraph.hpp. Each float3 is followed by a
// float so HLSL packs the pair into one register -- 4 registers per light.
struct PunctualLight
{
    float3 Position;      // world space; meaningless for a directional light
    float  Range;         // 0 == unlimited, else the point/spot cutoff distance

    float3 Direction;     // world space, normalised, the way the light TRAVELS
    float  Intensity;     // lux for directional, candela for point and spot

    float3 Color;         // linear RGB
    int    Type;          // 0 directional, 1 point, 2 spot

    float  InnerConeCos;  // spot only: cos(innerConeAngle)
    float  OuterConeCos;  // spot only: cos(outerConeAngle), <= InnerConeCos
    float2 Pad;
};

#define PBR_MAX_LIGHTS 4

cbuffer PBR_Constants : register(b0)
{
    float3   PBR_EyePosition            : packoffset(c0);
    float4x4 PBR_World                  : packoffset(c1);
    float3x3 PBR_WorldInverseTranspose  : packoffset(c5);
    float4x4 PBR_WorldViewProj          : packoffset(c8);
    float4x4 PBR_PrevWorldViewProj      : packoffset(c12);

    // The scene's punctual lights. Only PBR_Lights[0 .. PBR_LightCount-1] hold
    // a light; the remainder are zeroed, so iterating past the count costs
    // nothing but reads black. Lights the asset has beyond PBR_MAX_LIGHTS are
    // dropped on the CPU side and reported in the hierarchy UI.
    PunctualLight PBR_Lights[PBR_MAX_LIGHTS] : packoffset(c16);

    float3 PBR_ConstantAlbedo           : packoffset(c32);   // Constant values if not a textured effect
    float  PBR_ConstantMetallic         : packoffset(c33.x);
    float  PBR_ConstantRoughness        : packoffset(c33.y);

    int PBR_NumRadianceMipLevels        : packoffset(c33.z);
    int PBR_LightCount                  : packoffset(c33.w);

    // Size of render target
    float PBR_TargetWidth               : packoffset(c34.x);
    float PBR_TargetHeight              : packoffset(c34.y);
};

// Joint matrices for the node being drawn, one per joint of its skin, already
// including the inverse bind matrix and with the node's own world transform
// divided out -- so skinning here produces a position in the node's local
// space, exactly where an unskinned POSITION already is. PBR_World then applies
// as usual.
//
// Mirrors JointMatrixConstants in Engine/SceneGraph.hpp; PBR_MAX_JOINTS must
// equal kMaxJointMatrices there. Entries past the skin's joint count are
// identity, so a stray index is harmless.
#define PBR_MAX_JOINTS 128

cbuffer JointMatrices : register(b2)
{
    float4x4 SkinJoints[PBR_MAX_JOINTS];
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

    // Whether this primitive supplied a TANGENT stream. Geometry rather than
    // material, but it arrives in this buffer because this buffer is per
    // primitive. Normal mapping needs it: glTF says to derive tangents from the
    // UVs when the attribute is absent.
    int    Mat_HasTangents                  : packoffset(c6.z);

    // Set only when this primitive's node references a skin AND the primitive
    // supplies both JOINTS_0 and WEIGHTS_0. A primitive with joints but no
    // weights is left unskinned deliberately: every weight would read as zero
    // and collapse the mesh onto the origin.
    int    Mat_IsSkinned                    : packoffset(c6.w);
};

#define NoTextureRootSignature \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | " \
              "DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
              "CBV(b0), " \
              "DescriptorTable ( SRV(t0, numDescriptors = 5), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "CBV(b1), " \
              "CBV(b2, visibility = SHADER_VISIBILITY_VERTEX)"

// The five glTF metallic-roughness texture slots, in the order the SRV
// descriptor table stages them. Slots whose material has no texture receive
// the white fallback, so every one is always safe to sample.
Texture2D<float4> BaseColorTexture         : register(t0);
Texture2D<float4> MetallicRoughnessTexture : register(t1);
Texture2D<float4> NormalTexture            : register(t2);
Texture2D<float4> OcclusionTexture         : register(t3);
Texture2D<float4> EmissiveTexture          : register(t4);
sampler Sampler : register(s0);

float DistributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a; // 1.6*10^-3
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = M_PI * denom * denom;

    return nom / denom;
}

float GeometrySmith(float NdotV, float NdotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float ggx1 = NdotV / (NdotV * (1.0 - k) + k);
    float ggx2 = NdotL / (NdotL * (1.0 - k) + k);

    return ggx1 * ggx2;
}

float3 FresnelSchlick(float cosTheta, float3 F0, float3 F90=float3(1.0, 1.0, 1.0)) {
    return F0 + (F90 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Vertex shader: self-created quad.
[RootSignature(NoTextureRootSignature)]
VSOutput VSGBuffer(VSInput v)
{
    VSOutput vout;

    // Skinning runs first, in the node's local space, so everything below is
    // unchanged from the unskinned case.
    float3 Position = v.Position;
    float3 Normal = v.Normal;
    float3 Tangent = v.Tangent.xyz;

    if (Mat_IsSkinned > 0)
    {
        // glTF says the weights sum to 1; dividing by the actual total tolerates
        // an exporter that left them unnormalised, and the guard keeps a vertex
        // with no weight at all where it started rather than sending it to the
        // origin.
        float TotalWeight = v.Weights.x + v.Weights.y + v.Weights.z + v.Weights.w;
        if (TotalWeight > 1e-5)
        {
            // min() rather than trusting the data: an index past the array would
            // otherwise read outside the constant buffer.
            uint4 Indices = min(v.Joints, PBR_MAX_JOINTS - 1);

            float4x4 SkinMatrix =
                v.Weights.x * SkinJoints[Indices.x] +
                v.Weights.y * SkinJoints[Indices.y] +
                v.Weights.z * SkinJoints[Indices.z] +
                v.Weights.w * SkinJoints[Indices.w];
            SkinMatrix /= TotalWeight;

            Position = mul(float4(v.Position, 1.0), SkinMatrix).xyz;
            // The 3x3 part, not its inverse transpose. Joint matrices are
            // rigid in the common case, where the two agree; a non-uniformly
            // scaled bone would want the inverse transpose here.
            Normal = mul(v.Normal, (float3x3)SkinMatrix);
            Tangent = mul(v.Tangent.xyz, (float3x3)SkinMatrix);
        }
    }

    vout.NDCPosition = mul(float4(Position, 1.0), PBR_WorldViewProj);
    float4 WorldPosition = mul(float4(Position, 1.0), PBR_World);
    vout.WorldPosition = WorldPosition.xyz / WorldPosition.w;
    vout.TexCoord = v.TexCoord;
    vout.WorldNormal = mul(Normal, PBR_WorldInverseTranspose);

    // Tangents take the world matrix's linear part, NOT the inverse transpose
    // the normal uses: a tangent lies along the surface, so it transforms like a
    // difference of positions, while a normal has to stay perpendicular to one.
    // Under a mirroring (negative determinant) transform the bitangent
    // handedness in w flips; nothing here compensates for that yet.
    vout.WorldTangent = float4(mul(Tangent, (float3x3)PBR_World), v.Tangent.w);
    return vout;
}


//--------------------------------------------------------------------------------------
// Pixel shader: pass-through
// Reference from https://docs.vulkan.org/tutorial/latest/Building_a_Simple_Engine/Loading_Models/05_pbr_rendering.html
// and https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/cc27919cacbb235d2f58a0c0203387efce9375f8/source/Renderer/shaders/pbr.frag

// --- the G-buffer -----------------------------------------------------------
// Deferred shading splits the pixel shader in two: the geometry pass resolves
// each pixel's material inputs and writes them here, and the lighting pass
// reads them back once per pixel instead of once per pixel per primitive.
//
// Four targets plus depth. The layout is driven by what the shading below
// actually consumes -- adding a channel is cheap, but a target nothing reads
// costs bandwidth on every pixel of every frame:
//
//   0  R8G8B8A8_UNORM      base colour rgb, alpha in a
//   1  R16G16B16A16_FLOAT  world-space shading normal (after normal mapping)
//   2  R8G8B8A8_UNORM      r metallic, g roughness, b occlusion
//   3  R11G11B10_FLOAT     emissive rgb
//      D32_FLOAT           depth, which the lighting pass turns back into a
//                          world position rather than storing one
//
// Normals get 16-bit float rather than being packed into 8-bit: a UNORM normal
// bands visibly on a smooth specular highlight, which is the one place this
// renderer is going to look at them closely. Octahedral encoding into two
// channels would be the cheaper fix if bandwidth ever matters.
//
// Target 0 stays linear UNORM rather than _SRGB. An sRGB target would spend its
// 8 bits where the eye can see them, which is the better choice for albedo --
// but it would also apply a transfer function on write and undo it on read,
// silently, which is exactly the kind of thing that is maddening to debug when
// a base colour does not match its texture. Flip both this and the format in
// DeferredRenderer.cc if the banding in dark albedo ever shows.
//
// Mirrors kGBufferTargetCount and the formats in Engine/DeferredRenderer.cc.
struct GBufferOutput
{
    float4 BaseColor : SV_Target0;
    float4 Normal    : SV_Target1;
    float4 Material  : SV_Target2;
    float4 Emissive  : SV_Target3;
};

[RootSignature(NoTextureRootSignature)]
GBufferOutput PSGBuffer(VSOutput pin)
{
    // Since GLTF adopt metallic-roughness workflow to model realistic light transportation, we currently implement this lighting model
    // TODO we currently use if-condition in the shader to determine whether the texture is available, future development may require using #ifdef to accelerate it
    float4 BaseColor = BaseColorTexture.Sample(Sampler, pin.TexCoord);
    float Metallic = Mat_MetallicFactor;
    float Roughness = Mat_RoughnessFactor;
    float3 Normal = normalize(pin.WorldNormal);
    if (Mat_HasMetallicRoughnessTexture > 0) {
        // TODO need to separate Sampler
        float2 MetallicRoughness = MetallicRoughnessTexture.Sample(Sampler, pin.TexCoord).bg;
        Metallic *= MetallicRoughness.x;
        Roughness *= MetallicRoughness.y;
    }
    if (Mat_HasNormalTexture > 0) {
        // TODO currently share the same texcoord, object may be other texcoord
        // Renormalize [0-1] to [-1, 1]
        float3 TangentNormal = NormalTexture.Sample(Sampler, pin.TexCoord).xyz * 2.0 - 1.0;
        float3 Tangent = normalize(pin.WorldTangent.xyz);
        float3 Bitangent = normalize(cross(Normal, Tangent)) * pin.WorldTangent.w;
        float3x3 TBN = float3x3(Tangent, Bitangent, Normal);
        Normal = normalize(mul(TangentNormal, TBN));
    }

    float AmbientOcclusion = 1.0;
    if (Mat_HasOcclusionTexture > 0) {
        AmbientOcclusion = OcclusionTexture.Sample(Sampler, pin.TexCoord).r;
    }

    float3 Emissive = float3(0.0, 0.0, 0.0);
    if (Mat_HasEmissiveTexture > 0) {
        Emissive = EmissiveTexture.Sample(Sampler, pin.TexCoord).rgb;
    }

    GBufferOutput gbuffer;
    gbuffer.BaseColor = BaseColor;
    // Already unit length: both branches above normalise. The lighting pass
    // normalises again on read, because 16-bit floats do not store a unit
    // vector exactly.
    gbuffer.Normal = float4(Normal, 0.0);
    gbuffer.Material = float4(Metallic, Roughness, AmbientOcclusion, 0.0);
    gbuffer.Emissive = float4(Emissive, 0.0);
    return gbuffer;
}


//--------------------------------------------------------------------------------------
// Deferred lighting
//
// One fullscreen draw that reads the G-buffer and runs the shading that used to
// run per primitive. The body below is the second half of the old PSStraight,
// unchanged -- only where its inputs come from has changed.
//
// Registers do not overlap the geometry pass's on purpose. Both entry points
// live in one file, so a cbuffer at b0 and another at b0 would collide at
// declaration even though no single entry point uses both. The root signature
// decides what is actually bound, so the numbers only have to be distinct.
//--------------------------------------------------------------------------------------

// How many lights one deferred pass can light a pixel with. The forward path's
// PBR_MAX_LIGHTS stays at 4 because PBR_Constants is per node -- a 64-light
// array there would be uploaded once per node per frame. This buffer is
// uploaded once per frame, so it can afford the room.
//
// Mirrors kMaxDeferredLights in Engine/DeferredRenderer.hpp.
#define DEFERRED_MAX_LIGHTS 64

cbuffer DeferredLightingConstants : register(b3)
{
    // Turns a pixel's depth back into a world position. Rebuilding the
    // position from depth rather than storing it in the G-buffer saves a
    // full RGBA16F target, at the cost of this matrix and a divide.
    float4x4 Light_InvViewProj      : packoffset(c0);

    float3   Light_EyePosition      : packoffset(c4);
    int      Light_Count            : packoffset(c4.w);

    float    Light_TargetWidth      : packoffset(c5.x);
    float    Light_TargetHeight     : packoffset(c5.y);
    float2   Light_Pad              : packoffset(c5.z);

    PunctualLight Light_Lights[DEFERRED_MAX_LIGHTS] : packoffset(c6);
};

// The G-buffer, as written by PSGBuffer above. Read with Load rather than
// Sample: the lighting pass runs at exactly the G-buffer's resolution, so
// there is nothing to filter and no sampler to bind.
Texture2D<float4> GBufferBaseColor : register(t5);
Texture2D<float4> GBufferNormal    : register(t6);
Texture2D<float4> GBufferMaterial  : register(t7);
Texture2D<float4> GBufferEmissive  : register(t8);
Texture2D<float>  GBufferDepth     : register(t9);

#define DeferredLightingRootSignature \
    "RootFlags(DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
              "CBV(b3), " \
              "DescriptorTable ( SRV(t5, numDescriptors = 5), visibility = SHADER_VISIBILITY_PIXEL )"

struct LightingVSOutput
{
    float4 NDCPosition : SV_Position;
};

// A fullscreen triangle from three vertices and no vertex buffer. Bigger than
// the screen, so the parts outside it are clipped rather than rasterised; a
// quad would need two triangles and would rasterise the diagonal twice.
[RootSignature(DeferredLightingRootSignature)]
LightingVSOutput VSLighting(uint vertexId : SV_VertexID)
{
    LightingVSOutput vout;
    const float2 xy = float2((vertexId == 1) ? 3.0 : -1.0,
                             (vertexId == 2) ? 3.0 : -1.0);
    vout.NDCPosition = float4(xy, 0.0, 1.0);
    return vout;
}

[RootSignature(DeferredLightingRootSignature)]
float4 PSLighting(LightingVSOutput pin) : SV_Target0
{
    const int3 pixel = int3(pin.NDCPosition.xy, 0);

    // Nothing was drawn here, so leave whatever cleared the target. Geometry
    // that genuinely lands on the far plane is discarded with it, which is the
    // usual trade for not carrying a separate coverage channel.
    const float Depth = GBufferDepth.Load(pixel);
    if (Depth >= 1.0)
    {
        discard;
    }

    const float4 BaseColor = GBufferBaseColor.Load(pixel);
    const float3 Normal = normalize(GBufferNormal.Load(pixel).xyz);
    const float4 MaterialSample = GBufferMaterial.Load(pixel);
    const float Metallic = MaterialSample.r;
    const float Roughness = MaterialSample.g;
    const float AmbientOcclusion = MaterialSample.b;
    const float3 Emissive = GBufferEmissive.Load(pixel).rgb;

    // Pixel centre -> NDC -> world. The y flip is the usual one between a
    // top-left pixel origin and a bottom-left NDC origin. mul(vector, matrix)
    // matches the row-vector convention the rest of this file uses.
    const float2 ScreenUV = (pin.NDCPosition.xy + 0.5) /
                            float2(Light_TargetWidth, Light_TargetHeight);
    const float4 ClipPosition = float4(ScreenUV.x * 2.0 - 1.0,
                                       1.0 - ScreenUV.y * 2.0, Depth, 1.0);
    const float4 WorldPosition4 = mul(ClipPosition, Light_InvViewProj);
    const float3 WorldPosition = WorldPosition4.xyz / WorldPosition4.w;

    float3 ViewDirection = normalize(Light_EyePosition - WorldPosition); // Object to camera direction
    float3 ViewReflectionDirection = reflect(-ViewDirection, Normal); // The reflection direction of Object to camera direction, reflect require incident direction (Therefore -ViewDirection)

    // BaseReflectivity represents reflectance at normal incidence, or a 0-degree angle straight-on in Fresnel-Schlick approximation, denoted as F_0 in its equation
    float3 DialectricBaseReflectivity = float3(0.04, 0.04, 0.04); // 4% base reflectivity for non-metal objects

    float3 Radiance = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < Light_Count; ++i) {
        float3 LightPos = Light_Lights[i].Position;
        float3 LightColor = Light_Lights[i].Color;

        float3 LightVector = LightPos - WorldPosition;
        float Distance = length(LightVector);
        float3 LightDirection = normalize(LightVector); // Object to light direction
        float Attenuation = 1.0 / (Distance * Distance);
        float3 PerLightRadiance = 10.0 * LightColor * Attenuation;


        float3 HalfwayDirection = normalize(LightDirection + ViewDirection);
        float NormalDotLightDirection = max(0.0, dot(Normal, LightDirection));
        float NormalDotViewDirection = max(0.0, dot(Normal, ViewDirection));
        float NormalDotHalfwayDirection = max(0.0, dot(Normal, HalfwayDirection));
        float HalfwayDirectionDotViewDirection = max(0.0, dot(HalfwayDirection, ViewDirection));

        // Specular BRDF
        // The Distribution term serves to represent microfacet face orientation statistics given roughness and the dot result of normal-halfway
        float DistributionTerm = DistributionGGX(NormalDotHalfwayDirection, Roughness);
        // The Geometry term serves to represent self-occlusion statistics, it's accounted for microfacet's face occlude other faces and how occluded faces receive other face's reflection (masking and shadowing)
        float GeometryTerm = GeometrySmith(NormalDotViewDirection, NormalDotLightDirection, Roughness);
        // The Fresnel term serves to represent transmission statistics
        float3 DielectricFresnelTerm = FresnelSchlick(HalfwayDirectionDotViewDirection, DialectricBaseReflectivity);
        float3 MetalFresnelTerm = FresnelSchlick(HalfwayDirectionDotViewDirection, BaseColor.rgb);

        // Cook-Torance GGX distribution to model long-tail specular highlight
        // TODO need to look at https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/cc27919cacbb235d2f58a0c0203387efce9375f8/source/Renderer/shaders/brdf.glsl#L159
        // and https://github.com/KhronosGroup/glTF-Sample-Renderer/blob/cc27919cacbb235d2f58a0c0203387efce9375f8/source/Renderer/shaders/pbr.frag#L376C28-L376C38
        float3 Numerator = DistributionTerm * GeometryTerm;
        float Denominator = 4.0 * NormalDotViewDirection * NormalDotLightDirection + 0.0001;
        float3 SpecularIntensity = PerLightRadiance * NormalDotLightDirection * Numerator; // TODO our term differ in * (Numerator / Denominator) and Geometry term's smith calculation
        float3 DiffuseIntensity = PerLightRadiance * NormalDotLightDirection * (BaseColor.rgb / M_PI);

        float3 MetallicBRDF = MetalFresnelTerm * SpecularIntensity;
        float3 DielectricBRDF = lerp(DiffuseIntensity, SpecularIntensity, DielectricFresnelTerm);

        Radiance += lerp(MetallicBRDF, DielectricBRDF, Metallic);
    }
    float3 Ambient = float3(0.03, 0.03, 0.03) * BaseColor.rgb * AmbientOcclusion;
    float3 OutputColor = Ambient + Radiance;
    OutputColor += Emissive;

    // TODO Gamma correction

    return float4(OutputColor, BaseColor.a);
}
