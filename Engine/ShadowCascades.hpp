#pragma once

// Cascaded shadow maps: splitting the camera frustum by depth and fitting one
// orthographic light projection to each slice.
//
// Pure arithmetic, no D3D12, so //Engine:scenegraph-smoke can assert on it.
// That matters more here than usual: nearly every way of getting this wrong
// still produces shadows, just shadows that are in the wrong place, or that
// crawl when the camera moves, or that vanish at a cascade boundary.
//
// Two choices worth stating up front, because they are what make the result
// stable rather than merely correct:
//
//   * Each slice is fitted with a bounding SPHERE, not a bounding box. A box
//     fitted to the slice's corners changes size as the camera rotates, which
//     makes the shadow texels change size with it and the edges visibly crawl.
//     A sphere is rotation-invariant, so the fit only changes when the camera
//     moves, and the snapping below handles that.
//   * The fitted centre is SNAPPED to whole shadow-map texels. Without it the
//     projection slides by a fraction of a texel every frame and every shadow
//     edge shimmers. This is the single most visible difference between a
//     cascade implementation that looks finished and one that does not.
//   * A fit is HELD until the camera's slice no longer fits inside it, rather
//     than recomputed from scratch every frame. Snapping stabilises the texel
//     grid but not the matrix: the centre's component along the light, the
//     fitted radius and the light's eye position all slide continuously, so
//     measured on the Bistro every cascade's matrix changed after 1 mm of
//     camera motion or 0.01 degrees of yaw, with texels of 1.6 to 24 cm. That
//     is fine for drawing but fatal for caching anything, so each fit is padded
//     by `fitMargin` and reused while the slice stays within it.

#include <cstdint>

#include <directx/d3d12.h>

#include <DirectXMath.h>
#include <SimpleMath.h>

namespace NeuralModelIntegrateTestbed {

// Four is the usual answer: enough that the near cascade is tight without the
// atlas or the per-cascade culling cost getting out of hand.
constexpr int kMaxShadowCascades = 4;
// --- the sun's direction, as two angles ------------------------------------
//
// A direction has two degrees of freedom, so two Euler angles name it exactly
// and a third would be a roll about the direction itself, changing nothing.
// Both of these are about the direction the light TRAVELS -- what glTF stores
// and what the shader consumes, the light node's -Z axis in world space -- so
// a sun in the sky shines downwards and has a NEGATIVE pitch.
//
//   pitch  elevation of that direction: 0 horizontal, -90 straight down.
//   yaw    measured from -Z, so (0, 0) is glTF's default (0, 0, -1).
//
// The yaw zero is the camera panel's, so an angle read off one means the same
// thing in the other.
DirectX::SimpleMath::Vector3 SunDirectionFromEuler(float pitchDegrees,
                                                   float yawDegrees);

// The inverse. Needed because enabling an override has to start from wherever
// the sun already is, or the light jumps and the scene appears to break on a
// checkbox.
void SunEulerFromDirection(const DirectX::SimpleMath::Vector3& direction,
                           float& outPitchDegrees, float& outYawDegrees);



// The atlas is created TYPELESS so the lighting pass can read the same bits the
// shadow pass writes -- the same pairing DeviceResources needs for the scene
// depth buffer, and for the same reason: D3D12 will not put a shader resource
// view on a resource created as D32_FLOAT.
//
// 32-bit rather than D16_UNORM. The far cascade covers tens of metres across a
// 2048-texel tile, and 16 bits of depth over that range puts the quantisation
// close to the depth bias, which shows up as acne that no bias setting clears.
constexpr DXGI_FORMAT kShadowResourceFormat = DXGI_FORMAT_R32_TYPELESS;
constexpr DXGI_FORMAT kShadowDepthFormat = DXGI_FORMAT_D32_FLOAT;
constexpr DXGI_FORMAT kShadowShaderResourceFormat = DXGI_FORMAT_R32_FLOAT;

struct ShadowCascadeSetup {
    int cascadeCount = kMaxShadowCascades;

    // How far from the camera shadows are cast at all. Deliberately NOT the
    // camera's far plane: the Bistro's is 325 units, and spreading four
    // cascades over that spends most of the resolution on geometry too far
    // away to see a shadow edge on.
    float shadowDistance = 100.0f;

    // Blends a uniform split (0) with a logarithmic one (1). A uniform split
    // wastes the near cascade on a sliver of the view; a fully logarithmic one
    // makes the far cascade enormous. The usual compromise is in between.
    float splitLambda = 0.75f;

    // One cascade's edge, in texels. The atlas holds the cascades in a square
    // grid, so it is this times two on each side for four cascades.
    std::uint32_t resolution = 2048;

    // How much larger than the slice a fit is made, as a fraction of the
    // slice's radius. The margin is what the camera moves through before the
    // cascade has to be refitted, so it is the whole reason a cached shadow
    // map ever hits; 0 reproduces the old behaviour of refitting every frame.
    //
    // It is paid for in resolution: the same texel count covers a sphere this
    // much bigger, so shadows are coarser by the same fraction. 0.12 buys
    // roughly 2 m of travel on the Bistro's nearest cascade for 12% coarser
    // texels.
    float fitMargin = 0.12f;
};

struct ShadowCascade {
    // World space straight to this cascade's clip space.
    DirectX::SimpleMath::Matrix viewProjection;

    // The slice's far edge as a distance from the camera, which is what the
    // lighting pass compares a pixel's view depth against to pick a cascade.
    float splitDistance = 0.0f;

    // Where this cascade sits in the atlas, as a scale and offset applied to
    // the [0,1] shadow-map coordinate: uv * xy + zw.
    DirectX::SimpleMath::Vector4 atlasScaleOffset;

    // The world-space size of one texel, for the lighting pass's normal offset
    // and for sizing a depth bias in units that mean something.
    float texelWorldSize = 0.0f;

    // The padded world-space sphere this fit covers. A later call tests a new
    // slice against it to decide whether the fit can be reused.
    DirectX::SimpleMath::Vector3 fittedCentre;
    float fittedRadius = 0.0f;

    // False when this cascade's projection was carried over unchanged from the
    // `previous` fit, which is what says a cached shadow map for it is still
    // valid. True on a refit, and on the first fit.
    bool refitted = true;
};

struct ShadowCascades {
    ShadowCascade cascades[kMaxShadowCascades];
    int count = 0;

    // Side of the square atlas in texels.
    std::uint32_t atlasResolution = 0;

    // What the fits were made for. A change in any of it invalidates every
    // held fit, because the held spheres were chosen against these.
    DirectX::SimpleMath::Vector3 lightDirection;
    DirectX::SimpleMath::Vector3 sceneMin;
    DirectX::SimpleMath::Vector3 sceneMax;

    bool AnyRefitted() const {
        for (int i = 0; i < count; ++i) {
            if (cascades[i].refitted) {
                return true;
            }
        }
        return false;
    }
};

// Fits `setup.cascadeCount` cascades to the camera described by `view` and
// `projection`.
//
// `lightDirection` is the direction the light TRAVELS, matching
// ShaderLight::direction. `sceneMin`/`sceneMax` bound everything that could
// cast: the fitted depth range is taken from them rather than from the slice,
// because a caster behind the slice still has to be in the light's view or its
// shadow simply will not exist.
//
// Returns a count of 0 when there is nothing sensible to fit -- a degenerate
// light direction, a zero shadow distance, or a camera whose near plane is
// already past the shadow distance.
// `previous`, when given, is the last fit: any cascade whose new slice still
// fits inside the sphere that fit covered is carried over verbatim and reports
// `refitted == false`. Pass nullptr to fit everything from scratch.
ShadowCascades ComputeShadowCascades(const DirectX::SimpleMath::Matrix& view,
                                     const DirectX::SimpleMath::Matrix& projection,
                                     const DirectX::SimpleMath::Vector3& lightDirection,
                                     const DirectX::SimpleMath::Vector3& sceneMin,
                                     const DirectX::SimpleMath::Vector3& sceneMax,
                                     const ShadowCascadeSetup& setup,
                                     const ShadowCascades* previous = nullptr);

// One orthographic view covering the WHOLE scene from the light, for BAKING
// rather than for drawing.
//
// Cascades exist because a camera sees near things in more detail than far
// ones. A lightmap is indexed by the surface, not by the camera, so it has no
// near and far: the depth it is baked against has to cover every caster at
// once, in one projection, at one resolution.
//
// Fitted to the scene's bounding sphere, so the result does not depend on how
// the box happens to be oriented relative to the light. Returns false for a
// degenerate light direction or empty bounds, having written nothing.
bool SceneSunViewProjection(const DirectX::SimpleMath::Vector3& lightDirection,
                            const DirectX::SimpleMath::Vector3& sceneMin,
                            const DirectX::SimpleMath::Vector3& sceneMax,
                            std::uint32_t resolution,
                            DirectX::SimpleMath::Matrix& outViewProjection,
                            float& outTexelWorldSize);

// Where each cascade lands in the atlas, for a square grid. Exposed because
// the shadow pass needs the same rectangle as a viewport that the lighting pass
// needs as a UV transform, and the two must not be allowed to disagree.
void ShadowCascadeAtlasRect(int index, int cascadeCount, std::uint32_t resolution,
                            std::uint32_t& outX, std::uint32_t& outY,
                            std::uint32_t& outSize, std::uint32_t& outAtlasResolution);

}  // namespace NeuralModelIntegrateTestbed
