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

    // The LIGHTMAP UV set, glTF's TEXCOORD_1. A parameterisation of the
    // surface that does not overlap itself, which is what a baked per-surface
    // result has to be addressed by. TexCoord above cannot serve: it is a
    // texturing UV and on a converted scene it tiles. Reads as 0 where the
    // asset has no second set -- branch on Mat_HasLightmap, not on its value.
    float2 TexCoord1 : TexCoord1;
};

struct VSOutput
{
    float4 NDCPosition : SV_Position;
    float3 WorldPosition : Position;
    float2 TexCoord : TexCoord0;
    float3 WorldNormal : Normal;
    // xyz in world space, w carried through from the attribute.
    float4 WorldTangent : Tangent;
    // Already transformed into the atlas, so the pixel shader samples it
    // directly. Zero when this primitive has no chart.
    float2 LightmapUv : LightmapUv;
};

struct PSInput
{
    float4 Diffuse  : COLOR0;
    float4 Specular : COLOR1;
};

static const float GAMMA = 2.2;
static const float INV_GAMMA = 1.0 / GAMMA;
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

    // KHR_materials_pbrSpecularGlossiness, in its authored form. When
    // Mat_IsSpecularGlossiness is set, Mat_BaseColorFactor carries the
    // extension's diffuseFactor and the metallic-roughness texture slot
    // carries its specularGlossinessTexture.
    float3 Mat_SpecularFactor               : packoffset(c7);
    float  Mat_GlossinessFactor             : packoffset(c7.w);
    int    Mat_IsSpecularGlossiness         : packoffset(c8.x);

    // Where this primitive's chart sits in the lightmap atlas. A zero scale
    // means nothing was baked for it -- an asset with no second UV set, a
    // primitive that moves, or a chart the pack could not place -- and the
    // shadow map is then the only answer.
    float4 Mat_LightmapScaleOffset          : packoffset(c9);
};

#define NoTextureRootSignature \
    "RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | " \
              "DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
              "CBV(b0), " \
              "DescriptorTable ( SRV(t0, numDescriptors = 5), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "DescriptorTable ( Sampler(s0), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "DescriptorTable ( SRV(t5), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "StaticSampler(s4, filter = FILTER_MIN_MAG_MIP_LINEAR, " \
              "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, " \
              "visibility = SHADER_VISIBILITY_PIXEL), " \
              "CBV(b1), " \
              "CBV(b2, visibility = SHADER_VISIBILITY_VERTEX)"

// The baked result, sampled by the G-buffer pass: that is the only pass that
// has the surface's lightmap UV, so the term is resolved there and handed to
// the lighting pass through the G-buffer rather than re-derived per light.
Texture2D<float> LightmapAtlas : register(t5);
SamplerState     LightmapSampler : register(s4);

// The five glTF metallic-roughness texture slots, in the order the SRV
// descriptor table stages them. Slots whose material has no texture receive
// the white fallback, so every one is always safe to sample.
Texture2D<float4> BaseColorTexture         : register(t0);
Texture2D<float4> MetallicRoughnessTexture : register(t1);
Texture2D<float4> NormalTexture            : register(t2);
Texture2D<float4> OcclusionTexture         : register(t3);
Texture2D<float4> EmissiveTexture          : register(t4);
sampler Sampler : register(s0);

// --- KHR_materials_pbrSpecularGlossiness -> metallic-roughness --------------
//
// The archived extension's own conversion, from
// extensions/2.0/Archived/KHR_materials_pbrSpecularGlossiness/examples/
// convert-between-workflows/js/three.pbrUtilities.js. Run per PIXEL rather than
// per material, because the solve reads metalness out of the specular COLOUR
// and most of these materials keep that colour in a texture: their factor is
// (1,1,1) meaning "use the texture", which on its own solves to fully metallic.
//
// It is done here, at G-buffer write time, so the lighting pass only ever sees
// metallic-roughness and needs to know nothing about the extension.

static const float kDielectricSpecular = 0.04;
static const float kSpecGlossEpsilon = 1e-6;

// sqrt(0.299 r^2 + 0.587 g^2 + 0.114 b^2), as three.colorExtensions.js defines
// it. Note it is NOT the Rec.709 luminance used elsewhere for exposure: the
// weights are Rec.601 and it is applied to the SQUARES.
float PerceivedBrightness(float3 colour) {
    return sqrt(dot(float3(0.299, 0.587, 0.114), colour * colour));
}

// Solves the quadratic that relates a dielectric's diffuse+specular pair to the
// single base colour a metallic-roughness material would need.
float SolveMetallic(float diffuse, float specular, float oneMinusSpecularStrength) {
    if (specular < kDielectricSpecular) {
        return 0.0;
    }
    const float a = kDielectricSpecular;
    const float b = diffuse * oneMinusSpecularStrength / (1.0 - kDielectricSpecular) +
                    specular - 2.0 * kDielectricSpecular;
    const float c = kDielectricSpecular - specular;
    const float D = max(0.0, b * b - 4.0 * a * c);
    return clamp((-b + sqrt(D)) / (2.0 * a), 0.0, 1.0);
}

float DistributionGGX(float NdotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH2 = NdotH * NdotH;

    float nom = a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = M_PI * denom * denom;

    return nom / denom;
}

// Smith joint GGX VISIBILITY, as glTF-Sample-Renderer's brdf.glsl V_GGX.
//
// Visibility, not geometry: this already carries the 1 / (4 NdotL NdotV) that
// the Cook-Torrance denominator would otherwise supply, which is why the
// specular term below is just V * D with nothing underneath it.
//
//     GGXV = NdotL * sqrt(NdotV^2 (1 - a^2) + a^2)
//     GGXL = NdotV * sqrt(NdotL^2 (1 - a^2) + a^2)
//     V    = 0.5 / (GGXV + GGXL)
//
// `a` is alphaRoughness == perceptualRoughness^2, so a^2 is roughness^4 --
// the same convention DistributionGGX above already uses for its own a2, and
// the reason both take the perceptual roughness straight from the G-buffer.
//
// The guard matters: at a grazing angle both terms go to zero together and the
// quotient is 0/0.
float VisibilityGGX(float NdotV, float NdotL, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;

    float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);

    float GGX = GGXV + GGXL;
    if (GGX > 0.0) {
        return 0.5 / GGX;
    }
    return 0.0;
}

// Schlick-GGX geometry term with UE4's k = (roughness + 1)^2 / 8.
//
// Superseded by VisibilityGGX above and kept only for comparison. Two things
// made it the wrong half of the pair: it returns G rather than G/(4 NdotL NdotV),
// so it needs the Cook-Torrance denominator applied separately, and even with
// that denominator it diverges from the Smith joint form away from normal
// incidence -- measured against V_GGX * D_GGX it matches to 1.00x head-on,
// 0.66x at 60 degrees and 0.31x at grazing, so it darkens silhouettes.
float GeometrySmith(float NdotV, float NdotL, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;

    float GGXV = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float GGXL = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);

    float GGX = GGXV + GGXL;
    if (GGX > 0.0)
    {
        return 0.5 / GGX;
    }
    return 0.0;
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

    // Into the atlas, so the pixel shader has nothing to work out. Zero scale
    // leaves this at zero, and the pixel shader branches on the scale rather
    // than on the UV.
    vout.LightmapUv = v.TexCoord1 * Mat_LightmapScaleOffset.xy +
                      Mat_LightmapScaleOffset.zw;
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
// What alpha a BLEND material has to beat to be written at all.
//
// Deferred holds one surface per pixel, so true blending cannot happen in this
// pass -- it needs the lit colour of whatever is behind, which does not exist
// yet. MASK is exact (clip at the material's own cutoff); BLEND is not
// expressible, and both available answers are wrong in a different way:
//
//   render it opaque        glass stays visible, but a leaf card becomes a
//                           solid rectangle
//   clip at 0.5             the leaf is right, and a wine bottle at alpha 0.3
//                           disappears entirely
//
// 0.1 splits them, because the two cases do not actually overlap: foliage alpha
// maps are essentially binary, so a leaf's empty region is at or near 0, while
// glass sits well above. Fully transparent pixels are discarded and translucent
// ones are written opaque.
//
// The real fix is a forward pass after the lighting resolve, sharing this
// pass's depth buffer. Until then this is a knob, and it is here rather than in
// a constant buffer because it is one policy for the whole renderer.
static const float kDeferredBlendCutoff = 0.1;

struct GBufferOutput
{
    float4 BaseColor : SV_Target0;
    float4 Normal    : SV_Target1;
    float4 Material  : SV_Target2;
    float4 Emissive  : SV_Target3;
};

[RootSignature(NoTextureRootSignature)]
GBufferOutput PSGBuffer(VSOutput pin, bool IsFrontFace : SV_IsFrontFace)
{
    // Since GLTF adopt metallic-roughness workflow to model realistic light transportation, we currently implement this lighting model
    // TODO we currently use if-condition in the shader to determine whether the texture is available, future development may require using #ifdef to accelerate it
    float4 BaseColor = BaseColorTexture.Sample(Sampler, pin.TexCoord);

    // Specular-glossiness materials are converted here, so everything below
    // this point -- and the whole lighting pass -- is metallic-roughness only.
    float SpecGlossMetallic = 0.0;
    float SpecGlossRoughness = 1.0;
    if (Mat_IsSpecularGlossiness > 0)
    {
        // diffuseFactor rides in Mat_BaseColorFactor, and the base colour slot
        // holds diffuseTexture. Unlike the metallic-roughness path below, the
        // factor IS applied: the conversion is a function of the actual diffuse
        // and specular colours, so dropping either factor changes the metallic
        // it solves for, not just the brightness.
        const float3 Diffuse = Mat_BaseColorFactor.rgb * BaseColor.rgb;

        // The metallic-roughness slot holds specularGlossinessTexture when this
        // flag is set: RGB specular colour, A glossiness.
        float3 Specular = Mat_SpecularFactor;
        float Glossiness = Mat_GlossinessFactor;
        if (Mat_HasMetallicRoughnessTexture > 0)
        {
            const float4 SpecularGlossiness =
                MetallicRoughnessTexture.Sample(Sampler, pin.TexCoord);
            Specular *= SpecularGlossiness.rgb;
            Glossiness *= SpecularGlossiness.a;
        }

        const float OneMinusSpecularStrength =
            1.0 - max(Specular.r, max(Specular.g, Specular.b));
        SpecGlossMetallic = SolveMetallic(PerceivedBrightness(Diffuse),
                                         PerceivedBrightness(Specular),
                                         OneMinusSpecularStrength);
        SpecGlossRoughness = 1.0 - Glossiness;

        const float3 BaseColorFromDiffuse =
            Diffuse * (OneMinusSpecularStrength / (1.0 - kDielectricSpecular) /
                       max(1.0 - SpecGlossMetallic, kSpecGlossEpsilon));
        const float3 BaseColorFromSpecular =
            (Specular - kDielectricSpecular * (1.0 - SpecGlossMetallic)) *
            (1.0 / max(SpecGlossMetallic, kSpecGlossEpsilon));
        BaseColor = float4(
            clamp(lerp(BaseColorFromDiffuse, BaseColorFromSpecular,
                       SpecGlossMetallic * SpecGlossMetallic), 0.0, 1.0),
            BaseColor.a * Mat_BaseColorFactor.a);
    }

    // Alpha coverage, decided before anything else is computed so a cut-out
    // pixel costs only the base colour read. clip() discards the pixel, which
    // also stops it writing depth -- so the lighting pass never sees it and
    // the forward overlays can be seen through the hole.
    if (Mat_AlphaMode == 1)
    {
        // MASK: glTF says the pixel is fully opaque above the cutoff and fully
        // transparent at or below it.
        clip(BaseColor.a - Mat_AlphaCutoff);
    }
    else if (Mat_AlphaMode == 2)
    {
        clip(BaseColor.a - kDeferredBlendCutoff);
    }

    float Metallic = (Mat_IsSpecularGlossiness > 0) ? SpecGlossMetallic
                                                   : Mat_MetallicFactor;
    float Roughness = (Mat_IsSpecularGlossiness > 0) ? SpecGlossRoughness
                                                     : Mat_RoughnessFactor;
    float3 Normal = normalize(pin.WorldNormal);

    // A double-sided surface's back face has a normal pointing away from the
    // viewer, so lighting it unflipped makes a leaf or a curtain read as lit
    // from behind. glTF requires the flip, and it has to happen before the
    // normal map is applied because the TBN below is built from this normal.
    //
    // SV_IsFrontFace follows the rasterizer's winding, and the double-sided PSO
    // in GLTFAdapter::PreparePSO is the single-sided one with CullMode set to
    // NONE, so the two agree on which way round front is.
    if (Mat_DoubleSided > 0 && !IsFrontFace)
    {
        Normal = -Normal;
    }
    // Skipped for a spec-gloss material: that slot holds its
    // specularGlossinessTexture, which the block above has already consumed.
    // Reading it again as B = metallic, G = roughness would feed specular blue
    // in as metalness.
    if (Mat_HasMetallicRoughnessTexture > 0 && Mat_IsSpecularGlossiness == 0) {
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
    // The baked sun visibility for this surface point, resolved here because
    // this is the only pass that knows where the point is in its own chart.
    // The lighting pass reads it back out of Material.w.
    //
    // 1.0 -- fully lit -- where nothing was baked, so an unbaked scene is lit
    // by the shadow map alone rather than plunged into darkness.
    float BakedSunVisibility = 1.0;
    if (Mat_LightmapScaleOffset.x > 0.0)
    {
        BakedSunVisibility = LightmapAtlas.Sample(LightmapSampler, pin.LightmapUv);
    }
    gbuffer.Material = float4(Metallic, Roughness, AmbientOcclusion,
                              BakedSunVisibility);
    gbuffer.Emissive = float4(Emissive, 0.0);
    return gbuffer;
}



//--------------------------------------------------------------------------------------
// Shadow depth
//
// The cascade passes render depth only, from the light, into one tile of the
// shadow atlas each. The vertex shader is VSGBuffer itself: it takes its clip
// position from PBR_WorldViewProj, so substituting the cascade's light
// view-projection for the camera's is the whole change -- no second copy of the
// skinning, and the joint matrices and material constants come along for free.
//
// A pixel shader is still needed, for one reason: alpha-masked foliage. Without
// the clip, a leaf card casts the shadow of a solid rectangle, which is far more
// obvious than it sounds on a scene with as many plants as the Bistro.
//--------------------------------------------------------------------------------------

[RootSignature(NoTextureRootSignature)]
VSOutput VSShadow(VSInput v)
{
    return VSGBuffer(v);
}

[RootSignature(NoTextureRootSignature)]
void PSShadow(VSOutput pin)
{
    // Only the alpha is read, and only when the material has cut-outs: an
    // opaque primitive needs no pixel shader work at all.
    if (Mat_AlphaMode != 0)
    {
        float Alpha = BaseColorTexture.Sample(Sampler, pin.TexCoord).a;
        if (Mat_IsSpecularGlossiness > 0)
        {
            Alpha *= Mat_BaseColorFactor.a;
        }
        clip(Alpha - ((Mat_AlphaMode == 1) ? Mat_AlphaCutoff : kDeferredBlendCutoff));
    }
}

//--------------------------------------------------------------------------------------
// Baking a shadow into the surface
//
// A shadow map is indexed by the light's view; this is indexed by the
// receiver's surface. The vertex shader therefore rasterises into the LIGHTMAP
// UV rather than into the camera -- the triangle is drawn where it lives in its
// own parameterisation, and the pixel shader asks, for that point on the
// surface, whether the sun reaches it.
//
// Being independent of the camera is the whole point: the answer is a property
// of the surface, so it is computed once and never again.

cbuffer LightmapBakeConstants : register(b4)
{
    // The whole scene from the sun in one orthographic projection. Not a
    // cascade: a lightmap texel anywhere can need any caster, so there is no
    // near or far to slice by.
    float4x4 LM_SunViewProjection;
    // Where this primitive's chart sits in the atlas: uv * xy + zw.
    float4   LM_ChartScaleOffset;
    // Subtracted from the comparison depth, in the sun projection's units.
    float    LM_DepthBias;
    // Along the surface normal, in world units, before projecting. The same
    // trick the cascades use and for the same reason: it clears acne without
    // detaching the shadow from its caster.
    float    LM_NormalBias;
    // One sun-depth texel in world units, for scaling the normal offset.
    float    LM_SunTexelSize;
    // One sun-depth texel in UV, i.e. 1 / resolution, for stepping the kernel.
    // Passed rather than derived: the shader has no other way to know how big
    // the map it is sampling is.
    float    LM_SunUvTexel;
};

Texture2D<float>       LightmapSunDepth : register(t13);
SamplerComparisonState LightmapSunSampler : register(s3);

#define LightmapBakeRootSignature \
    "RootFlags(DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
              "CBV(b0), " \
              "CBV(b4), " \
              "DescriptorTable ( SRV(t13), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "StaticSampler(s3, filter = FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, " \
              "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, " \
              "comparisonFunc = COMPARISON_LESS_EQUAL, " \
              "visibility = SHADER_VISIBILITY_PIXEL)"

struct LightmapBakeVSOutput
{
    float4 Position : SV_Position;
    float3 World    : WorldPosition;
    float3 Normal   : Normal;
};

[RootSignature(LightmapBakeRootSignature)]
LightmapBakeVSOutput VSLightmapBake(VSInput v)
{
    LightmapBakeVSOutput output;

    // The chart's place in the atlas, turned into clip space. The y flip is
    // the usual one: UV grows downwards, clip space grows upwards.
    const float2 AtlasUv = v.TexCoord1 * LM_ChartScaleOffset.xy + LM_ChartScaleOffset.zw;
    output.Position = float4(AtlasUv.x * 2.0 - 1.0, 1.0 - AtlasUv.y * 2.0, 0.0, 1.0);

    // Only static geometry is baked, so there is no skinning here: a skinned
    // primitive's surface is not where its authored vertices say it is, which
    // is exactly why it is excluded from the bake.
    output.World = mul(float4(v.Position, 1.0), PBR_World).xyz;
    output.Normal = normalize(mul(v.Normal, (float3x3)PBR_WorldInverseTranspose));
    return output;
}

[RootSignature(LightmapBakeRootSignature)]
float PSLightmapBake(LightmapBakeVSOutput pin) : SV_Target0
{
    const float3 Normal = normalize(pin.Normal);

    // Offset along the normal before projecting, scaled by a sun-depth texel:
    // a texel covers more world space the more obliquely it meets the surface,
    // and this is what stops a surface shadowing itself.
    const float3 Offset = pin.World + Normal * (LM_NormalBias * LM_SunTexelSize);

    float4 Clip = mul(float4(Offset, 1.0), LM_SunViewProjection);
    if (Clip.w <= 0.0)
    {
        return 1.0;
    }
    const float3 Ndc = Clip.xyz / Clip.w;
    const float2 SunUv = float2(Ndc.x * 0.5 + 0.5, 0.5 - Ndc.y * 0.5);
    if (any(SunUv < 0.0) || any(SunUv > 1.0) || Ndc.z > 1.0)
    {
        // Outside the baked sun view: nothing recorded a caster there, so the
        // honest answer is lit rather than an invented shadow.
        return 1.0;
    }

    const float Reference = Ndc.z - LM_DepthBias;

    // A wider kernel than the runtime cascades use. A bake is paid for once,
    // so there is no reason to be stingy, and a lightmap texel is coarser than
    // a shadow texel -- 5x5 over the sun map is roughly the footprint of one
    // lightmap texel, which is the right amount of averaging.
    float Sum = 0.0;
    [unroll] for (int y = -2; y <= 2; ++y)
    {
        [unroll] for (int x = -2; x <= 2; ++x)
        {
            const float2 Tap = SunUv + float2(x, y) * LM_SunUvTexel;
            Sum += LightmapSunDepth.SampleCmpLevelZero(LightmapSunSampler, Tap,
                                                       Reference);
        }
    }
    return Sum / 25.0;
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
#define DEFERRED_MAX_LIGHTS 128

// Mirrors kMaxShadowCascades in Engine/ShadowCascades.hpp.
#define SHADOW_MAX_CASCADES 4

// Mirrors ShadowCascadeConstants in Engine/DeferredRenderer.hpp. Six registers
// each: the matrix, the atlas rectangle, and two scalars with their padding.
struct ShadowCascadeData
{
    float4x4 ViewProjection;
    // uv * xy + zw maps a cascade's own [0,1] coordinate into the atlas.
    float4   AtlasScaleOffset;
    // The slice's far edge as a distance from the camera.
    float    SplitDistance;
    // One texel of this cascade, in world units.
    float    TexelWorldSize;
    float2   Pad;
};

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

    // --- cascaded shadow maps -------------------------------------------
    // Appended after the light array rather than slotted in before it, which
    // is why the register numbers jump: 128 lights at four registers each
    // occupy c6 to c517.
    //
    // One struct per cascade so the lighting pass can index it with a value it
    // computed, rather than trying to dynamically index a component of a
    // float4.
    ShadowCascadeData Light_Cascades[SHADOW_MAX_CASCADES] : packoffset(c518);

    int   Light_ShadowCascadeCount    : packoffset(c542.x);
    // Subtracted from the comparison depth. Small, because the normal offset
    // below does most of the work.
    float Light_ShadowDepthBias       : packoffset(c542.y);
    // Multiplies a cascade's texel size to push the sample point along the
    // surface normal. This is what removes acne on slopes without the
    // peter-panning that a depth bias large enough to do the same would cause.
    float Light_ShadowNormalBias      : packoffset(c542.z);
    float Light_ShadowAtlasResolution : packoffset(c542.w);
};

// The G-buffer, as written by PSGBuffer above. Read with Load rather than
// Sample: the lighting pass runs at exactly the G-buffer's resolution, so
// there is nothing to filter and no sampler to bind.
Texture2D<float4> GBufferBaseColor : register(t5);
Texture2D<float4> GBufferNormal    : register(t6);
Texture2D<float4> GBufferMaterial  : register(t7);
Texture2D<float4> GBufferEmissive  : register(t8);
Texture2D<float>  GBufferDepth     : register(t9);

// The shadow atlas, as a comparison-sampled depth texture. t12 rather than t10
// or t11 because those belong to the skybox's environment map and the tonemap's
// scene colour: every entry point shares this file's declarations, so the
// numbers have to be distinct even though no root signature binds all of them.
Texture2D<float> ShadowAtlas : register(t12);
// A comparison sampler, so each tap is a hardware 2x2 percentage-closer filter
// rather than one depth fetch.
SamplerComparisonState ShadowSampler : register(s2);

#define DeferredLightingRootSignature \
    "RootFlags(DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
              "CBV(b3), " \
              "DescriptorTable ( SRV(t5, numDescriptors = 5), " \
              "SRV(t12, numDescriptors = 1), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "StaticSampler(s2, filter = FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT, " \
              "addressU = TEXTURE_ADDRESS_CLAMP, addressV = TEXTURE_ADDRESS_CLAMP, " \
              "comparisonFunc = COMPARISON_LESS_EQUAL, " \
              "visibility = SHADER_VISIBILITY_PIXEL)"

// Which cascade covers a pixel at this distance from the camera, or -1 when it
// is past the last one and therefore unshadowed.
//
// Written as a comparison chain rather than a loop so there is no dynamic
// indexing of a float4's components. Unused splits are zero, so a scene with
// fewer cascades falls through to -1 exactly as a pixel past the shadow
// distance does.
int SelectShadowCascade(float viewDepth)
{
    if (Light_ShadowCascadeCount > 0 && viewDepth <= Light_Cascades[0].SplitDistance) return 0;
    if (Light_ShadowCascadeCount > 1 && viewDepth <= Light_Cascades[1].SplitDistance) return 1;
    if (Light_ShadowCascadeCount > 2 && viewDepth <= Light_Cascades[2].SplitDistance) return 2;
    if (Light_ShadowCascadeCount > 3 && viewDepth <= Light_Cascades[3].SplitDistance) return 3;
    return -1;
}

// 1 fully lit, 0 fully shadowed.
float SampleShadow(int cascade, float3 WorldPosition, float3 Normal)
{
    if (cascade < 0)
    {
        return 1.0;
    }

    // Along the normal by about a texel. A pure depth bias large enough to
    // clear acne on a shallow slope also detaches the shadow from the foot of
    // whatever cast it; offsetting along the surface instead scales with how
    // coarse this particular cascade is.
    const float3 Offset =
        Normal * (Light_Cascades[cascade].TexelWorldSize * Light_ShadowNormalBias);
    float4 Clip = mul(float4(WorldPosition + Offset, 1.0),
                      Light_Cascades[cascade].ViewProjection);
    if (Clip.w <= 0.0)
    {
        return 1.0;
    }
    Clip.xyz /= Clip.w;
    // Outside this cascade's own box: nothing was rendered there, so claiming
    // shadow would be inventing it.
    if (any(abs(Clip.xy) > 1.0) || Clip.z < 0.0 || Clip.z > 1.0)
    {
        return 1.0;
    }

    const float2 CascadeUV = float2(Clip.x * 0.5 + 0.5, 0.5 - Clip.y * 0.5);
    const float4 Rect = Light_Cascades[cascade].AtlasScaleOffset;
    const float Texel = 1.0 / Light_ShadowAtlasResolution;
    // Taps are kept inside this cascade's tile. A 3x3 kernel at a tile edge
    // would otherwise read the neighbouring cascade's depths, which have
    // nothing to do with this pixel.
    const float2 TileMin = Rect.zw + Texel * 0.5;
    const float2 TileMax = Rect.zw + Rect.xy - Texel * 0.5;

    const float Depth = Clip.z - Light_ShadowDepthBias;
    float Sum = 0.0;
    [unroll] for (int y = -1; y <= 1; ++y)
    {
        [unroll] for (int x = -1; x <= 1; ++x)
        {
            const float2 UV = clamp(CascadeUV * Rect.xy + Rect.zw +
                                        float2(x, y) * Texel,
                                    TileMin, TileMax);
            Sum += ShadowAtlas.SampleCmpLevelZero(ShadowSampler, UV, Depth);
        }
    }
    return Sum / 9.0;
}

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
    // The baked sun visibility for this surface, resolved by the G-buffer pass
    // out of the lightmap. 1.0 where nothing was baked.
    const float BakedSunVisibility = MaterialSample.a;
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

    float3 ViewDirection = normalize(Light_EyePosition - WorldPosition); // Point to camera direction
    float3 ViewReflectionDirection = reflect(-ViewDirection, Normal); // The reflection direction of Point to camera direction, reflect require incident direction (Therefore -ViewDirection)

    // BaseReflectivity represents reflectance at normal incidence, or a 0-degree angle straight-on in Fresnel-Schlick approximation, denoted as F_0 in its equation
    float3 DialectricBaseReflectivity = float3(0.04, 0.04, 0.04); // 4% base reflectivity for non-metal objects

    // Cascaded shadows cover the directional light only. A point light would
    // need a cube map per light, and there are 96 of them in the Bistro.
    //
    // The cascade is chosen from distance along the view direction rather than
    // straight-line distance to the eye, so the boundaries are planes parallel
    // to the screen and do not bow towards the corners of the view.
    const float3 ViewForward = normalize(Light_EyePosition - WorldPosition);
    const float ViewDepth = distance(Light_EyePosition, WorldPosition);
    const int ShadowCascade = SelectShadowCascade(ViewDepth);

    // Two estimates of the same thing, combined with min rather than a
    // product. They see largely the SAME static occluders -- the bake and the
    // cascade are both looking at the building that casts the shadow -- so
    // multiplying them would darken one shadow twice. For a shared occluder
    // the lesser of two visibilities is the right answer, and where only one
    // of them sees an occluder at all, which is what happens for a moving
    // caster since those are never baked, the lesser is that one.
    //
    // Where the bake already says fully shadowed there is nothing the shadow
    // map can add, so its 3x3 comparison kernel is skipped outright -- which
    // is most of a baked scene's shadowed area.
    float DirectionalShadow = BakedSunVisibility;
    if (BakedSunVisibility > 0.0)
    {
        DirectionalShadow = min(BakedSunVisibility,
                                SampleShadow(ShadowCascade, WorldPosition, Normal));
    }

    float3 Radiance = float3(0.0, 0.0, 0.0);
    for (int i = 0; i < Light_Count; ++i) {
        float3 LightPos = Light_Lights[i].Position;
        float3 LightColor = Light_Lights[i].Color;
        float LightIntensity = Light_Lights[i].Intensity;

        float3 LightVector = LightPos - WorldPosition;
        float Distance = length(LightVector);
        float3 LightDirection = normalize(LightVector); // Point to light direction
        float RangeAttenuation = 1.0;
        // https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_lights_punctual/README.md#range-property
        if (Light_Lights[i].Type != 0) {
            // Not directional light
            if (Light_Lights[i].Range <= 0.0) {
                RangeAttenuation = 1.0 / (Distance * Distance);
            } else {
                RangeAttenuation = max(min(1.0 - pow(Distance / Light_Lights[i].Range, 4.0), 1.0), 0.0) / pow(Distance, 2.0);
            }
        } else {
            LightDirection = -Light_Lights[i].Direction;
        }
        float3 PerLightRadiance = RangeAttenuation * LightIntensity * LightColor;

        // The only shading change cascaded shadows make: the directional
        // light's radiance is attenuated by how much of it reaches this pixel.
        if (Light_Lights[i].Type == 0) {
            PerLightRadiance *= DirectionalShadow;
        }

        float3 HalfwayDirection = normalize(LightDirection + ViewDirection);
        float NormalDotLightDirection = max(0.0, dot(Normal, LightDirection));
        float NormalDotViewDirection = max(0.0, dot(Normal, ViewDirection));
        float NormalDotHalfwayDirection = max(0.0, dot(Normal, HalfwayDirection));
        float HalfwayDirectionDotViewDirection = max(0.0, dot(HalfwayDirection, ViewDirection));

        // Specular BRDF
        // The Distribution term serves to represent microfacet face orientation statistics given roughness and the dot result of normal-halfway
        float DistributionTerm = DistributionGGX(NormalDotHalfwayDirection, Roughness);
        // The Visibility term serves to represent self-occlusion statistics, it's accounted for microfacet's face occlude other faces and how occluded faces receive other face's reflection (masking and shadowing), divided by the Cook-Torrance 4 NdotL NdotV it already folds in
        float VisibilityTerm = VisibilityGGX(NormalDotViewDirection, NormalDotLightDirection, Roughness);
        // The Fresnel term serves to represent transmission statistics
        float3 DielectricFresnelTerm = FresnelSchlick(HalfwayDirectionDotViewDirection, DialectricBaseReflectivity);
        float3 MetalFresnelTerm = FresnelSchlick(HalfwayDirectionDotViewDirection, BaseColor.rgb);

        // Cook-Torrance GGX, as glTF-Sample-Renderer's BRDF_specularGGX:
        //
        //     Vis = V_GGX(NdotL, NdotV, alphaRoughness)
        //     D   = D_GGX(NdotH, alphaRoughness)
        //     return Vis * D
        //
        // There is deliberately no 4 NdotV NdotL denominator here. It used to
        // be computed into a `Denominator` local and then never applied, which
        // left the specular 4x too bright facing a light -- undamped on metals,
        // whose Fresnel is the base colour rather than 0.04.
        float3 BRDFSpecularGGX = VisibilityTerm * DistributionTerm;
        float3 SpecularIntensity = PerLightRadiance * NormalDotLightDirection * BRDFSpecularGGX;
        float3 DiffuseIntensity = PerLightRadiance * NormalDotLightDirection * (BaseColor.rgb / M_PI);

        float3 MetallicBRDF = MetalFresnelTerm * SpecularIntensity;
        float3 DielectricBRDF = lerp(DiffuseIntensity, SpecularIntensity, DielectricFresnelTerm);

        Radiance += lerp(DielectricBRDF, MetallicBRDF, Metallic);
    }
    float3 Ambient = float3(0.03, 0.03, 0.03) * BaseColor.rgb * AmbientOcclusion;
    float3 OutputColor = Ambient + Radiance;
    OutputColor += Emissive;

    // Linear, un-tonemapped, into the HDR scene colour target. The Reinhard
    // and the gamma that used to be here are in PSTonemap, which runs once over
    // the whole frame -- so the sky and the forward overlays get the same
    // transform instead of the sky being written raw and reading 2.4x too dark
    // against equivalently-lit geometry.
    return float4(OutputColor, BaseColor.a);
}


//--------------------------------------------------------------------------------------
// Skybox
//
// One fullscreen draw that turns each pixel back into a world-space view ray
// and looks that ray up in an equirectangular (latitude-longitude) environment
// map. Runs BEFORE the lighting resolve and owns the clear: every pixel is
// covered, so the lighting pass simply writes over the ones that have geometry
// and leaves the sky where it discards.
//
// Costs a full-screen texture fetch on pixels the lighting pass then paints
// over. The alternative is depth-testing this against the scene so it only
// fills the background, which saves a fetch per covered pixel and costs a
// depth-buffer transition; at this resolution the fetch is not worth the extra
// state. Registers are distinct from the other two passes for the same reason
// theirs are -- see the note above the lighting constants.
//--------------------------------------------------------------------------------------

cbuffer SkyboxConstants : register(b4)
{
    // Pixel -> world ray, exactly as the lighting pass rebuilds a position.
    float4x4 Sky_InvViewProj     : packoffset(c0);
    float3   Sky_EyePosition     : packoffset(c4);
    float    Sky_Exposure        : packoffset(c4.w);
    float    Sky_TargetWidth     : packoffset(c5.x);
    float    Sky_TargetHeight    : packoffset(c5.y);
    // Rotation about Y, in radians. An HDRI is captured facing some arbitrary
    // direction and almost never the one the scene wants.
    float    Sky_Yaw             : packoffset(c5.z);
    float    Sky_Pad             : packoffset(c5.w);
};

Texture2D<float4> SkyEnvironment : register(t10);
// Its own sampler rather than the geometry pass's: this one wraps in u, because
// longitude is periodic and a clamp would smear the seam across the sky.
SamplerState SkySampler : register(s1);

#define SkyboxRootSignature \
    "RootFlags(DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
              "CBV(b4), " \
              "DescriptorTable ( SRV(t10), visibility = SHADER_VISIBILITY_PIXEL ), " \
              "StaticSampler(s1, filter = FILTER_MIN_MAG_MIP_LINEAR, " \
              "addressU = TEXTURE_ADDRESS_WRAP, addressV = TEXTURE_ADDRESS_CLAMP, " \
              "visibility = SHADER_VISIBILITY_PIXEL)"

struct SkyboxVSOutput
{
    float4 NDCPosition : SV_Position;
};

[RootSignature(SkyboxRootSignature)]
SkyboxVSOutput VSSkybox(uint vertexId : SV_VertexID)
{
    SkyboxVSOutput vout;
    const float2 xy = float2((vertexId == 1) ? 3.0 : -1.0,
                             (vertexId == 2) ? 3.0 : -1.0);
    vout.NDCPosition = float4(xy, 0.0, 1.0);
    return vout;
}

[RootSignature(SkyboxRootSignature)]
float4 PSSkybox(SkyboxVSOutput pin) : SV_Target0
{
    // The same reconstruction the lighting pass does, at the far plane, giving
    // a point along the view ray rather than on a surface.
    const float2 ScreenUV = (pin.NDCPosition.xy + 0.5) /
                            float2(Sky_TargetWidth, Sky_TargetHeight);
    const float4 ClipPosition = float4(ScreenUV.x * 2.0 - 1.0,
                                       1.0 - ScreenUV.y * 2.0, 1.0, 1.0);
    const float4 FarPoint = mul(ClipPosition, Sky_InvViewProj);
    float3 Direction = normalize(FarPoint.xyz / FarPoint.w - Sky_EyePosition);

    // Yaw about Y, applied to the ray rather than to the image, so no
    // resampling and no seam.
    const float CosYaw = cos(Sky_Yaw);
    const float SinYaw = sin(Sky_Yaw);
    Direction = float3(Direction.x * CosYaw - Direction.z * SinYaw,
                       Direction.y,
                       Direction.x * SinYaw + Direction.z * CosYaw);

    // Equirectangular: longitude across u, latitude down v. u = 0.5 looks down
    // -Z, which is the direction a right-handed view matrix faces, so an
    // unrotated HDRI lines up with the camera's forward at yaw 0.
    const float2 EnvironmentUV = float2(
        atan2(Direction.x, -Direction.z) / (2.0 * M_PI) + 0.5,
        acos(clamp(Direction.y, -1.0, 1.0)) / M_PI);

    const float3 Radiance = SkyEnvironment.Sample(SkySampler, EnvironmentUV).rgb;
    return float4(Radiance * Sky_Exposure, 1.0);
}


//--------------------------------------------------------------------------------------
// Tonemap
//
// The frame's single display transform. Everything before it -- the sky, the
// deferred lighting resolve and the forward overlays -- writes linear radiance
// into an HDR scene colour target, and this reads that once and writes the
// back buffer.
//
// Having one of these is the point. With the transform at the end of
// PSLighting instead, the sky was written raw and the overlays were written
// raw, so three surfaces that should share a response had two different ones:
// at linear 0.18 the sky read 0.180 where geometry read 0.425.
//
// The ImGui pass deliberately runs AFTER this, straight onto the back buffer:
// a UI is authored in display space and must not be tonemapped.
//--------------------------------------------------------------------------------------

Texture2D<float4> SceneColor : register(t11);

#define TonemapRootSignature \
    "RootFlags(DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
              "DescriptorTable ( SRV(t11), visibility = SHADER_VISIBILITY_PIXEL )"

struct TonemapVSOutput
{
    float4 NDCPosition : SV_Position;
};

[RootSignature(TonemapRootSignature)]
TonemapVSOutput VSTonemap(uint vertexId : SV_VertexID)
{
    TonemapVSOutput vout;
    const float2 xy = float2((vertexId == 1) ? 3.0 : -1.0,
                             (vertexId == 2) ? 3.0 : -1.0);
    vout.NDCPosition = float4(xy, 0.0, 1.0);
    return vout;
}

[RootSignature(TonemapRootSignature)]
float4 PSTonemap(TonemapVSOutput pin) : SV_Target0
{
    // Load, not Sample: this runs at the scene colour target's own resolution,
    // so there is nothing to filter.
    float4 Scene = SceneColor.Load(int3(pin.NDCPosition.xy, 0));
    float3 OutputColor = Scene.rgb;

    // TODO Simple Reinhard Tonemapping
    OutputColor = OutputColor / (1.0 + OutputColor);

    // TODO Gamma correction
    OutputColor = pow(OutputColor, float3(INV_GAMMA, INV_GAMMA, INV_GAMMA));
    return float4(OutputColor, Scene.a);
}


//--------------------------------------------------------------------------------------
// Debug view
//
// Replaces the tonemap when the UI asks for an intermediate instead of the
// frame: one of the G-buffer targets, the depth buffer, the shadow atlas, the
// baked lightmap. It REPLACES rather than overlays because that is the version
// that cannot lie about what is in the buffer -- a thumbnail in a corner is
// resampled twice and the thing you are usually looking for is a single wrong
// texel.
//
// One shader for all of them, because what differs between a normal buffer and
// a roughness buffer is only which channels to show and how to map them onto
// [0,1]. The alternative is a pixel shader per target, all of which would be
// the same Sample call.
//
// Sample, not Load: a 4096-wide atlas has to be fitted into the window, which
// means filtering. Load is what the tonemap uses because it alone is guaranteed
// to be 1:1 with the back buffer. For a source that IS screen-sized the UV
// below lands exactly on a texel centre, so the filter returns that texel
// unchanged and this is still an honest 1:1 view.
//--------------------------------------------------------------------------------------

// Must match enum class DebugChannels in DeferredRenderer.hpp.
#define DEBUG_CHANNELS_TONEMAPPED  0  // rgb through the display transform
#define DEBUG_CHANNELS_COLOR       1  // rgb, gamma only -- a linear UNORM target
#define DEBUG_CHANNELS_NORMAL      2  // a signed direction, remapped to [0,1]
#define DEBUG_CHANNELS_RED         3
#define DEBUG_CHANNELS_GREEN       4
#define DEBUG_CHANNELS_BLUE        5
#define DEBUG_CHANNELS_ALPHA       6
#define DEBUG_CHANNELS_DEPTH       7  // 1/z from a perspective projection

cbuffer DebugViewConstants : register(b5)
{
    int   Debug_Channels     : packoffset(c0.x);
    float Debug_SourceWidth  : packoffset(c0.y);
    float Debug_SourceHeight : packoffset(c0.z);
    float Debug_Padding      : packoffset(c0.w);
    float Debug_TargetWidth  : packoffset(c1.x);
    float Debug_TargetHeight : packoffset(c1.y);
    // The window onto a scalar channel: what reads as black and what reads as
    // white. Identity at 0 and 1, which is what every colour view uses.
    float Debug_RangeMin     : packoffset(c1.z);
    float Debug_RangeMax     : packoffset(c1.w);
    // The camera's planes, for DEBUG_CHANNELS_DEPTH only. A depth buffer shown
    // raw is almost uniformly white: 1/z spends half its range on the first few
    // per cent of the view distance.
    float Debug_Near         : packoffset(c2.x);
    float Debug_Far          : packoffset(c2.y);
};

Texture2D<float4> DebugSource : register(t14);
SamplerState      DebugSampler : register(s5);

#define DebugViewRootSignature \
    "RootFlags(DENY_DOMAIN_SHADER_ROOT_ACCESS | " \
              "DENY_GEOMETRY_SHADER_ROOT_ACCESS | " \
              "DENY_HULL_SHADER_ROOT_ACCESS), " \
    "CBV(b5), " \
    "DescriptorTable ( SRV(t14), visibility = SHADER_VISIBILITY_PIXEL ), " \
    "StaticSampler(s5, " \
                  "filter = FILTER_MIN_MAG_MIP_LINEAR, " \
                  "addressU = TEXTURE_ADDRESS_CLAMP, " \
                  "addressV = TEXTURE_ADDRESS_CLAMP, " \
                  "visibility = SHADER_VISIBILITY_PIXEL )"

struct DebugViewVSOutput
{
    float4 NDCPosition : SV_Position;
};

[RootSignature(DebugViewRootSignature)]
DebugViewVSOutput VSDebugView(uint vertexId : SV_VertexID)
{
    DebugViewVSOutput vout;
    const float2 xy = float2((vertexId == 1) ? 3.0 : -1.0,
                             (vertexId == 2) ? 3.0 : -1.0);
    vout.NDCPosition = float4(xy, 0.0, 1.0);
    return vout;
}

[RootSignature(DebugViewRootSignature)]
float4 PSDebugView(DebugViewVSOutput pin) : SV_Target0
{
    // Fit the source into the target without stretching it. A square atlas
    // stretched to a 16:9 window is not just ugly -- it makes a chart look
    // like it has the wrong aspect, which is one of the things this view
    // exists to check.
    float2 uv = pin.NDCPosition.xy / float2(Debug_TargetWidth, Debug_TargetHeight);
    const float TargetAspect = Debug_TargetWidth / max(Debug_TargetHeight, 1.0);
    const float SourceAspect = Debug_SourceWidth / max(Debug_SourceHeight, 1.0);
    const float2 Fit = (TargetAspect > SourceAspect)
                           ? float2(TargetAspect / SourceAspect, 1.0)
                           : float2(1.0, SourceAspect / TargetAspect);
    uv = (uv - 0.5) * Fit + 0.5;
    if (any(uv < 0.0) || any(uv > 1.0))
    {
        // Letterbox. Not black: black is a legitimate value in every one of
        // these buffers, and the edge of the image is worth being able to see.
        return float4(0.05, 0.05, 0.06, 1.0);
    }

    const float4 Source = DebugSource.SampleLevel(DebugSampler, uv, 0.0);

    // One scalar, windowed. Shared by every single-channel view so that the two
    // sliders mean the same thing in all of them.
    float Scalar = 0.0;
    switch (Debug_Channels)
    {
        case DEBUG_CHANNELS_RED:   Scalar = Source.r; break;
        case DEBUG_CHANNELS_GREEN: Scalar = Source.g; break;
        case DEBUG_CHANNELS_BLUE:  Scalar = Source.b; break;
        case DEBUG_CHANNELS_ALPHA: Scalar = Source.a; break;
        case DEBUG_CHANNELS_DEPTH:
        {
            // Back to view-space metres, then linearly onto [0,1] across the
            // view distance -- so the window above is in units of "fraction of
            // the far plane" rather than of 1/z, and the far-plane clear is
            // white.
            //
            // A logarithmic mapping is the usual advice and it is WORSE here,
            // measurably: see the README. It spends its range on the gap
            // between the near plane and the scene, and this application sizes
            // the near plane to the scene at startup, so that gap is empty.
            const float Denominator =
                Debug_Far - Source.r * (Debug_Far - Debug_Near);
            const float ViewDepth =
                Debug_Near * Debug_Far / max(Denominator, 1e-6);
            Scalar = saturate(ViewDepth / max(Debug_Far, 1e-6));
            break;
        }
        default: break;
    }

    if (Debug_Channels >= DEBUG_CHANNELS_RED)
    {
        const float Window = max(Debug_RangeMax - Debug_RangeMin, 1e-6);
        const float Grey = saturate((Scalar - Debug_RangeMin) / Window);
        // No gamma. These channels are scalars, not radiance, and the number
        // the buffer holds is the number worth reading off the screen.
        return float4(Grey, Grey, Grey, 1.0);
    }

    if (Debug_Channels == DEBUG_CHANNELS_NORMAL)
    {
        // Renormalised first: the G-buffer stores an interpolated, normal-mapped
        // direction and a short one would otherwise read as grey-ish rather
        // than as the direction it points.
        const float3 Normal = normalize(Source.xyz);
        return float4(Normal * 0.5 + 0.5, 1.0);
    }

    float3 OutputColor = Source.rgb;
    if (Debug_Channels == DEBUG_CHANNELS_TONEMAPPED)
    {
        OutputColor = OutputColor / (1.0 + OutputColor);
    }
    // Gamma for both colour cases: the targets hold linear values and the
    // display is not linear. Without it a base colour of 0.5 shows as 0.5,
    // which is about twice as dark as the texture it came from looks.
    OutputColor = pow(saturate(OutputColor), float3(INV_GAMMA, INV_GAMMA, INV_GAMMA));
    return float4(OutputColor, 1.0);
}
