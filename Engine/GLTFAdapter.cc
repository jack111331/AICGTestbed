#include "pch.h"
#include "GLTFAdapter.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <cmath>
#include <cstring>
#include <string>
#include <sstream>
#include <memory>
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx12.h"
#include <directx/d3dx12.h>
#include <stb_image.h>
#include "Descriptors/DescriptorContext.hpp"
#include <filesystem>
#include <fstream>
#include "GBuffer.hpp"
#include "ShadowCascades.hpp"
#include <no_texture.h>

using namespace DirectX;

namespace NeuralModelIntegrateTestbed {

namespace {

// sRGB transfer function. Needed because downscaling averages pixels, and
// averaging sRGB-encoded values is not the same as averaging the light they
// represent -- the result comes out measurably dark. The GPU's GenerateMips
// gets this right for free on an _SRGB format (the SRV read decodes and the
// UAV write re-encodes); a CPU resize has to do it by hand.
float SrgbToLinear(unsigned char encoded) {
    const float value = static_cast<float>(encoded) / 255.0f;
    return value <= 0.04045f ? value / 12.92f
                             : std::pow((value + 0.055f) / 1.055f, 2.4f);
}

unsigned char LinearToSrgb(float linear) {
    const float clamped = std::min(1.0f, std::max(0.0f, linear));
    const float encoded = clamped <= 0.0031308f
                              ? clamped * 12.92f
                              : 1.055f * std::pow(clamped, 1.0f / 2.4f) - 0.055f;
    return static_cast<unsigned char>(encoded * 255.0f + 0.5f);
}

// One 2x2 box reduction of an RGBA8 image, in place. Alpha is always linear, so
// it averages directly whatever the colour channels do.
void HalveImage(std::vector<unsigned char> &pixels, int &width, int &height, bool srgb) {
    const int newWidth = std::max(1, width / 2);
    const int newHeight = std::max(1, height / 2);
    std::vector<unsigned char> out(static_cast<std::size_t>(newWidth) * newHeight * 4);
    for (int y = 0; y < newHeight; ++y) {
        const int y0 = std::min(y * 2, height - 1);
        const int y1 = std::min(y * 2 + 1, height - 1);
        for (int x = 0; x < newWidth; ++x) {
            const int x0 = std::min(x * 2, width - 1);
            const int x1 = std::min(x * 2 + 1, width - 1);
            const std::size_t source[4] = {
                (static_cast<std::size_t>(y0) * width + x0) * 4,
                (static_cast<std::size_t>(y0) * width + x1) * 4,
                (static_cast<std::size_t>(y1) * width + x0) * 4,
                (static_cast<std::size_t>(y1) * width + x1) * 4};
            const std::size_t destination =
                (static_cast<std::size_t>(y) * newWidth + x) * 4;
            for (int channel = 0; channel < 3; ++channel) {
                if (srgb) {
                    float sum = 0.0f;
                    for (const std::size_t offset : source) {
                        sum += SrgbToLinear(pixels[offset + channel]);
                    }
                    out[destination + channel] = LinearToSrgb(sum * 0.25f);
                } else {
                    int sum = 0;
                    for (const std::size_t offset : source) {
                        sum += pixels[offset + channel];
                    }
                    out[destination + channel] =
                        static_cast<unsigned char>((sum + 2) / 4);
                }
            }
            int alpha = 0;
            for (const std::size_t offset : source) {
                alpha += pixels[offset + 3];
            }
            out[destination + 3] = static_cast<unsigned char>((alpha + 2) / 4);
        }
    }
    pixels.swap(out);
    width = newWidth;
    height = newHeight;
}

// What a texture DirectXTK12 created for us costs, from its own description.
// Block compressed formats are the point: BC7 is one byte per pixel against
// RGBA8's four, and the DDS already carries its mips so nothing is generated.
std::uint64_t CompressedTextureBytes(const D3D12_RESOURCE_DESC& desc) {
    std::uint64_t bitsPerPixel = 32;
    switch (desc.Format) {
        case DXGI_FORMAT_BC1_UNORM:
        case DXGI_FORMAT_BC1_UNORM_SRGB:
        case DXGI_FORMAT_BC4_UNORM:
        case DXGI_FORMAT_BC4_SNORM:
            bitsPerPixel = 4;
            break;
        case DXGI_FORMAT_BC2_UNORM:
        case DXGI_FORMAT_BC2_UNORM_SRGB:
        case DXGI_FORMAT_BC3_UNORM:
        case DXGI_FORMAT_BC3_UNORM_SRGB:
        case DXGI_FORMAT_BC5_UNORM:
        case DXGI_FORMAT_BC5_SNORM:
        case DXGI_FORMAT_BC6H_UF16:
        case DXGI_FORMAT_BC6H_SF16:
        case DXGI_FORMAT_BC7_UNORM:
        case DXGI_FORMAT_BC7_UNORM_SRGB:
            bitsPerPixel = 8;
            break;
        default:
            break;
    }
    std::uint64_t total = 0;
    for (UINT level = 0; level < desc.MipLevels; ++level) {
        const std::uint64_t levelWidth = std::max<std::uint64_t>(1, desc.Width >> level);
        const std::uint64_t levelHeight = std::max<std::uint64_t>(1, desc.Height >> level);
        total += levelWidth * levelHeight * bitsPerPixel / 8;
    }
    return total;
}

// What one RGBA8 texture with a full mip chain costs in video memory. The
// chain adds a third again on top of mip 0, and nothing here is BC compressed.
std::uint64_t TextureBytes(int width, int height, bool hasMipChain) {
    std::uint64_t total = 0;
    int levelWidth = width;
    int levelHeight = height;
    for (;;) {
        total += static_cast<std::uint64_t>(levelWidth) * levelHeight * 4;
        if (!hasMipChain || (levelWidth == 1 && levelHeight == 1)) {
            break;
        }
        levelWidth = std::max(1, levelWidth / 2);
        levelHeight = std::max(1, levelHeight / 2);
    }
    return total;
}

}  // namespace

    fastgltf::Expected<fastgltf::Asset> GLTFAdapter::Initialize(const std::string &gltfFilepath) {
        #if defined(_DEBUG)
        Microsoft::WRL::ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();
        }
        #endif

        std::ifstream ifs(gltfFilepath);
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string gltfData = ss.str();
        auto data = fastgltf::GltfDataBuffer::FromBytes(
            reinterpret_cast<const std::byte*>(gltfData.data()), gltfData.size());
        if (data.error() != fastgltf::Error::None) {
            return data.error();
        }
        // KHR_lights_punctual is opt-in: fastgltf leaves Asset::lights empty
        // and drops every Node::lightIndex unless it is requested here.
        // MSFT_texture_dds: assets converted out of engine content frequently
        // ship ONLY block-compressed DDS, naming a .png in the texture's
        // ordinary source that was never shipped and the real file in the
        // extension. Without this the Lumberyard Bistro has no textures.
        //
        // KHR_materials_pbrSpecularGlossiness: archived, but FBX2glTF still
        // emits it for anything whose source engine worked that way. 234 of the
        // Bistro's 254 materials are specular-glossiness, and the base colour
        // lives inside the extension -- see the mapping in SceneGraph.cc.
        fastgltf::Extensions extensions = fastgltf::Extensions::MYLAB_generative |
                                          fastgltf::Extensions::KHR_lights_punctual |
                                          fastgltf::Extensions::MSFT_texture_dds |
                                          fastgltf::Extensions::KHR_materials_pbrSpecularGlossiness;
        fastgltf::Parser parser(extensions);
        std::filesystem::path absPath = std::filesystem::absolute(gltfFilepath);
        // LoadExternalImages is deliberately NOT set. fastgltf loads every image
        // eagerly under that flag and returns on the first file it cannot open,
        // which fails the whole parse -- and an MSFT_texture_dds asset has a
        // dead .png URI for every live .dds one (343 of each in the Bistro).
        // Images therefore arrive as sources::URI and PrepareImage opens the
        // ones it actually needs, skipping a missing file with a warning.
        //
        // LoadExternalBuffers stays: the geometry is one 91 MB .bin and there is
        // nothing optional about it.
        m_assetDirectory = absPath.parent_path();
        auto loadedGLTF = parser.loadGltfJson(data.get(), absPath.parent_path(),
                             fastgltf::Options::LoadGLBBuffers |
                             fastgltf::Options::LoadExternalBuffers);
        if (loadedGLTF.error() == fastgltf::Error::None) {
            m_gltf = std::move(loadedGLTF.get());
        } else {
            return loadedGLTF.error();
        }
        return fastgltf::Error::None;

    }

    void GLTFAdapter::PrepareBuffer(std::shared_ptr<DX::DeviceResources> deviceResources) {
        auto device = deviceResources->GetD3DDevice();

        // The ported descriptor classes reach for the device and the frame
        // number through this context, so it has to exist before any of them
        // are constructed. Idempotent: safe to call again after a device reset.
        Descriptors::Context().Initialize(device);

        // CPU-visible staging descriptors, grown a page at a time. 256 per page
        // is the upstream default; glTF images are allocated one descriptor at
        // a time so a page covers 256 textures before another is created.
        m_srvAllocator = std::make_unique<Descriptors::DescriptorAllocator>(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 256);

        ResourceUploadBatch resourceUpload(device);

        const CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);

        resourceUpload.Begin();
        for (const auto &buffer: m_gltf.buffers) {

            const auto desc = CD3DX12_RESOURCE_DESC::Buffer(buffer.byteLength);

            Microsoft::WRL::ComPtr<ID3D12Resource> createdBuffer;
            DX::ThrowIfFailed(device->CreateCommittedResource(&heapProperties,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                c_initialCopyTargetState,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(createdBuffer.GetAddressOf())));
            createdBuffer->SetName(L"GLTF buffer 1");
            buffers.push_back(createdBuffer);

            SetDebugObjectName(createdBuffer.Get(), L"ModelMeshPart");

            // Create staging buffer via GraphicsMemory and upload buffer data via memcpy, save allocation time using paging mechanism
            // https://github.com/microsoft/DirectXTK12/blob/f003171e0c6a30cc61444864ef756bd3cf2fc310/Src/GraphicsMemory.cpp#L126
            // TODO do we need to hold the SharedGraphicsResource even when the upload is finished?
            SharedGraphicsResource vertexBuffer = GraphicsMemory::Get(device).Allocate(buffer.byteLength, 16, GraphicsMemory::TAG_VERTEX);
            // access buffer.data (std::variant) using std::visitor like https://github.com/spnda/fastgltf/blob/main/examples/gl_viewer/gl_viewer.cpp#L516
            std::visit(fastgltf::visitor {
                [](const auto& arg) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::Vector& vector) {
                    memcpy(vertexBuffer.Memory(), vector.bytes.data(), buffer.byteLength);
                },
                [&](const fastgltf::sources::CustomBuffer& customBuffer) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::ByteView& byteView) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::Fallback& fallback) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const std::monostate& monostate) {
                    // TODO haven't implement loading from view
                    throw std::logic_error("Functionality not yet implemented!"); 
                },
                [&](const fastgltf::sources::URI& filePath) {
                    // TODO We may specify file offset
                    assert(filePath.uri.isLocalPath()); // We're only capable of loading local files.

                    const std::string path(filePath.uri.path().begin(), filePath.uri.path().end()); // Thanks C++.
                    std::ifstream meshBin(path.c_str(), std::ios::binary);
                    meshBin.read(reinterpret_cast<char*>(vertexBuffer.Memory()), filePath.fileByteOffset);
                    meshBin.close();
                },
                [&](const fastgltf::sources::Array& vector) {
                    memcpy(vertexBuffer.Memory(), vector.bytes.data(), buffer.byteLength);
                },
                [&](const fastgltf::sources::BufferView& view) {
                    // TODO haven't implement loading from view
                    auto& bufferView = m_gltf.bufferViews[view.bufferViewIndex];
                    const auto& buffer = m_gltf.buffers[bufferView.bufferIndex];
                    // Yes, we've already loaded every buffer into some GL buffer. However, with GL it's simpler
                    // to just copy the buffer data again for the texture. Besides, this is just an example.
                    std::visit(fastgltf::visitor {
                        // We only care about VectorWithMime here, because we specify LoadExternalBuffers, meaning
                        // all buffers are already loaded into a vector.
                        [](auto& arg) {},
                        [&](const fastgltf::sources::Array& vector) {
                            memcpy(vertexBuffer.Memory(), vector.bytes.data(), buffer.byteLength);
                        },

                    }, buffer.data);
                },
            }, buffer.data);

            resourceUpload.Upload(createdBuffer.Get(), vertexBuffer);

            resourceUpload.Transition(createdBuffer.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);

        }

        /*
        // Create constant buffer
        const auto desc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(PBREffectConstants));
        SharedGraphicsResource cBufferResource = GraphicsMemory::Get(device).AllocateConstant(sizeof(PBREffectConstants));

        DX::ThrowIfFailed(device->CreateCommittedResource(&heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            c_initialCopyTargetState,
            nullptr,
            IID_GRAPHICS_PPV_ARGS(cbuffer.ReleaseAndGetAddressOf())));
        cbuffer->SetName(L"GLTF cbuffer 1");
        PBREffectConstants pbrEffectConstant = {};
        const DirectX::XMFLOAT3 translation(0.0f, 0.0f, 1.0f);
        const DirectX::XMMATRIX modelMatrix = DirectX::XMMatrixTranslationFromVector(DirectX::XMLoadFloat3(&translation));
        const DirectX::XMMATRIX viewMatrix = DirectX::XMMatrixIdentity();
        pbrEffectConstant.worldViewProj = XMMatrixMultiply(XMMatrixPerspectiveFovLH(1.121f, 1280.0/720.f, 0.01f, 1.0f), XMMatrixMultiply(viewMatrix, modelMatrix));

        memcpy(cBufferResource.Memory(), &pbrEffectConstant, sizeof(PBREffectConstants));

        resourceUpload.Upload(cbuffer.Get(), cBufferResource);

        resourceUpload.Transition(cbuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        */

        auto uploadEndFuture = resourceUpload.End(deviceResources->GetCommandQueue());
        uploadEndFuture.wait();

        // The per-back-buffer, per-worker command allocators and the private
        // command list this class used to own are gone. The frame is recorded
        // into one command list now, because the render graph has to be able to
        // put a resource barrier between the geometry draws and the lighting
        // pass that reads what they wrote -- which is impossible across two
        // separately submitted lists without a fence.
        //
        // Multi-threaded recording was what the kWorkerThreadSize allocators
        // were for, and nothing ever used more than [0][0]. When it comes back
        // it belongs behind the graph, one list per pass, not one per class.
    }

    void GLTFAdapter::PrepareImage(std::shared_ptr<DX::DeviceResources> deviceResources) {
        m_textureBytes = 0;
        m_texturesDownscaled = 0;
        m_texturesOverBudget = 0;
        m_imagesMissing = 0;
        m_imagesFailed = 0;
        m_imagesUnused = 0;

        // Which images any texture actually points at. An MSFT_texture_dds
        // asset lists two images per texture -- the .dds it uses and a .png it
        // does not -- so loading every image in the file decodes the whole
        // second set to RGBA8 for nothing. Measured on the Lumberyard Bistro
        // that was 5.8 GB instead of 1.2 GB once the .png files were present.
        std::vector<bool> imageUsed(m_gltf.images.size(), false);
        for (const fastgltf::Texture& texture : m_gltf.textures) {
            const std::optional<std::size_t> source = TextureImageSource(texture);
            if (source.has_value() && source.value() < imageUsed.size()) {
                imageUsed[source.value()] = true;
            }
        }
        // Decide each image's colour space from the material slots that
        // reference it, before any texture resource is created.
        m_imageColorSpaces = ClassifyImageColorSpaces(m_gltf);
        {
            std::size_t srgbCount = 0;
            for (const ImageColorSpace space : m_imageColorSpaces.perImage) {
                if (space == ImageColorSpace::Srgb) {
                    ++srgbCount;
                }
            }
            std::printf("glTF images: %zu total, %zu sRGB, %zu linear\n",
                        m_imageColorSpaces.perImage.size(), srgbCount,
                        m_imageColorSpaces.perImage.size() - srgbCount);
            for (const std::size_t imageIdx : m_imageColorSpaces.conflicts) {
                std::printf("  warning: image %zu is used as both colour and linear "
                            "data; loaded as sRGB\n", imageIdx);
            }
        }

        // Light definitions are shared; how many actually get instanced depends
        // on the nodes, which BuildSceneGraph resolves later. Printed here
        // because an asset exported without KHR_lights_punctual is otherwise
        // indistinguishable from one deliberately lit by nothing.
        std::printf("glTF lights: %zu definition(s) in the asset\n", m_gltf.lights.size());

        auto device = deviceResources->GetD3DDevice();
        ResourceUploadBatch resourceUpload(device);

        const CD3DX12_HEAP_PROPERTIES heapProperties(D3D12_HEAP_TYPE_DEFAULT);

        resourceUpload.Begin();
        std::size_t imageIndex = 0;
        for (const auto &image: m_gltf.images) {
            // Create staging buffer via GraphicsMemory and upload buffer data via memcpy, save allocation time using paging mechanism
            // https://github.com/microsoft/DirectXTK12/blob/f003171e0c6a30cc61444864ef756bd3cf2fc310/Src/GraphicsMemory.cpp#L126
            // TODO do we need to hold the SharedGraphicsResource even when the upload is finished?
            int width = 0, height = 0, nrChannels = 0;
            std::vector<uint8_t> decodedData;

            // Declared out here because the DDS branch further down reads
            // them; the placeholder path below simply leaves them null.
            std::vector<uint8_t> fileBytes;
            const uint8_t* encoded = nullptr;
            std::size_t encodedSize = 0;

            // Unreferenced: give it a placeholder so the image indices the
            // materials resolved against stay aligned, and move on without
            // touching the file.
            if (imageIndex < imageUsed.size() && !imageUsed[imageIndex]) {
                ++m_imagesUnused;
                width = 1;
                height = 1;
                decodedData = {255, 0, 255, 255};
            } else {

            // Step one: get at the file's bytes, wherever they live. Images are
            // no longer pre-loaded by fastgltf (see Initialize), so most arrive
            // as a URI and are read here; a data: URI or a GLB still arrives as
            // bytes already.
            std::visit(fastgltf::visitor {
                [](const auto&) {},
                [&](const fastgltf::sources::URI& filePath) {
                    if (!filePath.uri.isLocalPath()) {
                        return;
                    }
                    // Relative to the .gltf, which is where glTF says a URI is
                    // resolved from.
                    const std::string relative(filePath.uri.path().begin(),
                                               filePath.uri.path().end());
                    const std::filesystem::path full = m_assetDirectory / relative;
                    std::ifstream file(full, std::ios::binary | std::ios::ate);
                    if (!file) {
                        // Not fatal. An MSFT_texture_dds asset names a .png
                        // source that was never shipped for every .dds it does
                        // ship, and nothing references those images -- so a
                        // missing file here is usually by design. The slot is
                        // left with no descriptor and the material reads the
                        // white fallback.
                        ++m_imagesMissing;
                        return;
                    }
                    const std::streamsize size = file.tellg();
                    if (size <= static_cast<std::streamsize>(filePath.fileByteOffset)) {
                        ++m_imagesMissing;
                        return;
                    }
                    file.seekg(static_cast<std::streamoff>(filePath.fileByteOffset));
                    fileBytes.resize(static_cast<std::size_t>(size) -
                                     static_cast<std::size_t>(filePath.fileByteOffset));
                    file.read(reinterpret_cast<char*>(fileBytes.data()),
                              static_cast<std::streamsize>(fileBytes.size()));
                    encoded = fileBytes.data();
                    encodedSize = fileBytes.size();
                },
                [&](const fastgltf::sources::Array& vector) {
                    encoded = reinterpret_cast<const uint8_t*>(vector.bytes.data());
                    encodedSize = vector.bytes.size();
                },
                [&](const fastgltf::sources::Vector& vector) {
                    encoded = reinterpret_cast<const uint8_t*>(vector.bytes.data());
                    encodedSize = vector.bytes.size();
                },
                [&](const fastgltf::sources::BufferView& view) {
                    const auto& bufferView = m_gltf.bufferViews[view.bufferViewIndex];
                    const auto& owner = m_gltf.buffers[bufferView.bufferIndex];
                    std::visit(fastgltf::visitor {
                        [](const auto&) {},
                        [&](const fastgltf::sources::Array& vector) {
                            encoded = reinterpret_cast<const uint8_t*>(vector.bytes.data()) +
                                      bufferView.byteOffset;
                            encodedSize = bufferView.byteLength;
                        },
                    }, owner.data);
                },
            }, image.data);

            }

            const ImageColorSpace colorSpace = m_imageColorSpaces.perImage[imageIndex];

            // Step two: DDS or not. A DDS is already block compressed with its
            // own mip chain, so DirectXTK12 uploads it as-is -- no decode, no
            // GenerateMips, and roughly a quarter of the video memory an RGBA8
            // copy of the same texture would take. That is what makes a scene
            // of this size fit at all.
            const bool isDds = encodedSize >= 4 && encoded != nullptr &&
                               encoded[0] == 'D' && encoded[1] == 'D' &&
                               encoded[2] == 'S' && encoded[3] == ' ';
            if (isDds) {
                Microsoft::WRL::ComPtr<ID3D12Resource> ddsTexture;
                // The cap becomes DirectXTK's maxsize, which drops top mip
                // levels rather than resampling -- the only way to shrink a
                // block-compressed texture without decompressing it.
                const std::size_t maxSize = m_maxTextureDimension;
                // A DDS carries its own format, and whether that format is the
                // _SRGB variant is up to whoever wrote the file. FORCE_SRGB
                // makes the decision ours, matching what the colour-space
                // classifier worked out from the slots the image is used in.
                const DDS_LOADER_FLAGS loadFlags =
                    (colorSpace == ImageColorSpace::Srgb) ? DDS_LOADER_FORCE_SRGB
                                                          : DDS_LOADER_DEFAULT;
                const HRESULT hr = CreateDDSTextureFromMemoryEx(
                    device, resourceUpload, encoded, encodedSize, maxSize,
                    D3D12_RESOURCE_FLAG_NONE, loadFlags,
                    ddsTexture.GetAddressOf());
                if (FAILED(hr)) {
                    std::fprintf(stderr,
                                 "image %zu: CreateDDSTextureFromMemoryEx failed 0x%08X "
                                 "(%zu bytes)\n",
                                 imageIndex, static_cast<unsigned>(hr), encodedSize);
                    std::fflush(stderr);
                }
                if (SUCCEEDED(hr)) {
                    const D3D12_RESOURCE_DESC desc = ddsTexture->GetDesc();
                    m_textureBytes += CompressedTextureBytes(desc);
                    if (maxSize != 0 && (desc.Width > maxSize || desc.Height > maxSize)) {
                        // maxsize could not be honoured; the file's smallest mip
                        // is already larger than the cap.
                        ++m_texturesOverBudget;
                    } else if (maxSize != 0) {
                        ++m_texturesDownscaled;
                    }
                    // No Transition here. CreateDDSTextureFromMemoryEx uploads
                    // AND transitions to PIXEL_SHADER_RESOURCE itself, so asking
                    // for COPY_DEST -> PIXEL_SHADER_RESOURCE afterwards names a
                    // StateBefore the resource is no longer in. The debug layer
                    // breaks on that, which is fatal without a debugger
                    // attached -- it crashed on the very first DDS image.
                    Descriptors::DescriptorAllocation ddsSrv = m_srvAllocator->Allocate(1);
                    if (ddsSrv.IsNull()) {
                        throw std::runtime_error("out of SRV descriptors for glTF images");
                    }
                    CreateShaderResourceView(device, ddsTexture.Get(),
                                             ddsSrv.GetDescriptorHandle());
                    m_imageDescriptors.push_back(std::move(ddsSrv));
                    images.push_back(ddsTexture);
                    SetDebugObjectName(ddsTexture.Get(), L"Image (DDS)");
                    ++imageIndex;
                    continue;
                }
                ++m_imagesFailed;
            }

            if (encoded != nullptr && !isDds) {
                unsigned char* data = stbi_load_from_memory(
                    reinterpret_cast<const stbi_uc*>(encoded),
                    static_cast<int>(encodedSize), &width, &height, &nrChannels, 4);
                if (data != nullptr) {
                    decodedData.assign(data, data + (static_cast<std::size_t>(width) *
                                                     height * 4));
                    stbi_image_free(data);
                } else {
                    // Was: assign(data, data + width*height*4) unconditionally,
                    // which read through a null pointer with uninitialised
                    // width and height the moment any image failed to decode.
                    ++m_imagesFailed;
                }
            }

            // An image with no usable pixels still needs an entry, so the image
            // indices the materials resolved against stay aligned. One opaque
            // magenta pixel, because a silently white texture is impossible to
            // tell from a correct one.
            if (decodedData.empty()) {
                width = 1;
                height = 1;
                decodedData = {255, 0, 255, 255};
            }

            // CD3DX12_RESOURCE_DESC::Tex2D defaults mipLevels to 0, which means
            // "allocate the full chain" -- a 1024x1024 image gets 11 mips. Only
            // subresource 0 is uploaded below, so the other 10 were left
            // uninitialised and sampled as black, which is why the model got
            // darker with distance as the sampler selected higher mips.
            //
            // Ask for the full chain only when the mips can actually be filled;
            // otherwise take a single level, so there is never an uninitialised
            // mip to sample either way.
            const DXGI_FORMAT imageFormat = (colorSpace == ImageColorSpace::Srgb)
                                                ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
                                                : DXGI_FORMAT_R8G8B8A8_UNORM;

            // The texture budget. Halving repeatedly rather than resampling to
            // the cap in one step, because a box reduction by two is exact and
            // needs no filter kernel -- and because the result is then the same
            // image the GPU's own mip chain would have produced at that level.
            if (m_maxTextureDimension != 0 && !decodedData.empty()) {
                const int cap = static_cast<int>(m_maxTextureDimension);
                bool reduced = false;
                while ((width > cap || height > cap) && (width > 1 || height > 1)) {
                    HalveImage(decodedData, width, height,
                               colorSpace == ImageColorSpace::Srgb);
                    reduced = true;
                }
                if (reduced) {
                    ++m_texturesDownscaled;
                }
            }
            const bool canGenerateMips =
                resourceUpload.IsSupportedForGenerateMips(imageFormat);
            const UINT16 mipLevels = canGenerateMips ? 0 : 1;
            m_textureBytes += TextureBytes(width, height, canGenerateMips);
            const auto desc = CD3DX12_RESOURCE_DESC::Tex2D(imageFormat, width, height, 1, mipLevels);

            // TODO get image data size from image.data
            Microsoft::WRL::ComPtr<ID3D12Resource> createdTexture;
            DX::ThrowIfFailed(device->CreateCommittedResource(&heapProperties,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                c_initialCopyTargetState,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(createdTexture.GetAddressOf())));
            createdTexture->SetName(L"GLTF image 1");
            // One page-allocated CPU descriptor per image, kept alongside it.
            // These are never bound directly -- BuildMaterialDescriptorTables
            // copies them into a shader-visible heap once the scene graph says
            // which material uses which -- so the allocator heap does not need
            // to be shader visible.
            Descriptors::DescriptorAllocation srv = m_srvAllocator->Allocate(1);
            if (srv.IsNull()) {
                throw std::runtime_error("out of SRV descriptors for glTF images");
            }
            CreateShaderResourceView(device, createdTexture.Get(), srv.GetDescriptorHandle());
            m_imageDescriptors.push_back(std::move(srv));
            images.push_back(createdTexture);

            SetDebugObjectName(createdTexture.Get(), L"Image");
            D3D12_SUBRESOURCE_DATA initData = {};
            initData.pData = decodedData.data();
            // TODO need further understanding
            initData.RowPitch = static_cast<LONG>(width * 4); // the physical size of a single texture row in bytes, since hardware prefer 256-byte for optimal copy performance, the texture in GPU will be padded to align 256 bytes
            initData.SlicePitch = static_cast<LONG>(height * initData.RowPitch);
            resourceUpload.Upload(createdTexture.Get(), 0, &initData, 1);

            resourceUpload.Transition(createdTexture.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);

            // Fills mips 1..N from mip 0 on the GPU. Must come after the
            // transition above: GenerateMips requires the resource to already be
            // in PIXEL_SHADER_RESOURCE, and it restores that state when done.
            //
            // sRGB cannot have a UAV created on it directly, so DirectXTK12
            // routes this through a non-sRGB alias internally; nothing extra is
            // needed here beyond the format check.
            if (canGenerateMips) {
                resourceUpload.GenerateMips(createdTexture.Get());
            }

            ++imageIndex;
        }

        // The fallback that fills unused material texture slots. Created here so
        // it rides along with the same upload batch as the glTF images.
        {
            const auto fallbackDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, 1);
            DX::ThrowIfFailed(device->CreateCommittedResource(&heapProperties,
                D3D12_HEAP_FLAG_NONE,
                &fallbackDesc,
                c_initialCopyTargetState,
                nullptr,
                IID_GRAPHICS_PPV_ARGS(m_fallbackTexture.GetAddressOf())));
            m_fallbackTexture->SetName(L"Material fallback (1x1 white)");

            m_fallbackTextureDescriptor = m_srvAllocator->Allocate(1);
            if (m_fallbackTextureDescriptor.IsNull()) {
                throw std::runtime_error("could not allocate the fallback texture descriptor");
            }
            CreateShaderResourceView(device, m_fallbackTexture.Get(),
                                     m_fallbackTextureDescriptor.GetDescriptorHandle());

            // Opaque white, so a slot the material leaves empty multiplies
            // through as a no-op for whatever shading reads it.
            static const uint8_t kWhite[4] = {0xFF, 0xFF, 0xFF, 0xFF};
            D3D12_SUBRESOURCE_DATA initData = {};
            initData.pData = kWhite;
            initData.RowPitch = 4;
            initData.SlicePitch = 4;
            resourceUpload.Upload(m_fallbackTexture.Get(), 0, &initData, 1);
            resourceUpload.Transition(m_fallbackTexture.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        }

        // Kept rather than removed with the rest of the load tracing. A scene
        // of this size spends minutes in here, and the difference between
        // "still uploading 1.2 GB of textures" and "hung" is not otherwise
        // visible. stderr, because a WINDOWS-subsystem binary has no stdout.
        std::fprintf(stderr,
                     "images: %zu loaded, %.1f MB of video memory, %zu unreferenced, "
                     "%zu file(s) absent, %zu failed to decode\n",
                     m_gltf.images.size() - m_imagesMissing - m_imagesFailed -
                         m_imagesUnused,
                     static_cast<double>(m_textureBytes) / 1048576.0, m_imagesUnused,
                     m_imagesMissing, m_imagesFailed);
        std::fflush(stderr);
        auto uploadEndFuture = resourceUpload.End(deviceResources->GetCommandQueue());
        uploadEndFuture.wait();

    }


    void GLTFAdapter::BuildSceneGraph() {
        // glTF's `scene` property picks the one to display; fall back to the
        // first if the asset does not name one.
        m_currentSceneIdx = m_gltf.defaultScene.value_or(0);
        m_sceneGraph.Build(m_gltf, m_currentSceneIdx, buffers, m_imageDescriptors.size());

        // What can be baked into a cached shadow map and what cannot. Worth
        // printing at load: a scene where most nodes move has nothing to gain
        // from the cache, and that is invisible from the frame rate alone.
        std::fprintf(stderr, "geometry: %zu static node(s), %zu that move\n",
                     m_sceneGraph.StaticGeometryNodeCount(),
                     m_sceneGraph.MovingGeometryNodeCount());
        std::fflush(stderr);

        // Skins degrade rather than fail -- a missing attribute or an unreadable
        // bind matrix still renders, just wrongly -- so report what was found
        // and anything suspect about it.
        const auto &skins = m_sceneGraph.Skins();
        if (!skins.empty()) {
            std::printf("glTF skins: %zu\n", skins.size());
            for (const auto &skin : skins) {
                std::printf("  \"%s\": %zu joints, inverse bind matrices %s\n",
                            skin.name.c_str(), skin.JointCount(),
                            skin.inverseBindMatricesLoaded ? "loaded"
                                                           : "substituted (identity)");
            }
        }
        const auto &animations = m_sceneGraph.Animations();
        if (!animations.empty()) {
            std::printf("glTF animations: %zu\n", animations.size());
            for (const auto &animation : animations) {
                std::printf("  \"%s\": %.3fs, %zu channels, %zu samplers",
                            animation.name.c_str(), animation.duration,
                            animation.channels.size(), animation.samplers.size());
                if (animation.skippedChannels != 0) {
                    std::printf(" (%zu channel(s) skipped: unsupported target)",
                                animation.skippedChannels);
                }
                std::printf("\n");
            }
        }
        for (const auto &diagnostic : m_sceneGraph.SkinDiagnostics()) {
            if (diagnostic.nodeIndex.has_value()) {
                std::printf("  warning: skin %zu, node %zu: %s\n", diagnostic.skinIndex,
                            diagnostic.nodeIndex.value(), ToString(diagnostic.issue));
            } else {
                std::printf("  warning: skin %zu: %s\n", diagnostic.skinIndex,
                            ToString(diagnostic.issue));
            }
        }
    }

    void GLTFAdapter::PreparePSO(std::shared_ptr<DX::DeviceResources> deviceResources) {
        auto device = deviceResources->GetD3DDevice();
        // One slot per VertexStream, in that order; must match VSInput in
        // shaders/no_texture.fx. Each attribute lives in its own buffer slot
        // rather than being interleaved, so AlignedByteOffset stays 0.
        D3D12_INPUT_ELEMENT_DESC inputElementDesc[kVertexStreamCount] = {
            { "Position", 0, DXGI_FORMAT_R32G32B32_FLOAT, static_cast<UINT>(VertexStream::Position), 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "TexCoord", 0, DXGI_FORMAT_R32G32_FLOAT,    static_cast<UINT>(VertexStream::TexCoord0), 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "Normal",   0, DXGI_FORMAT_R32G32B32_FLOAT, static_cast<UINT>(VertexStream::Normal),    0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            // 4 components: glTF TANGENT carries the bitangent handedness in w.
            { "Tangent",  0, DXGI_FORMAT_R32G32B32A32_FLOAT, static_cast<UINT>(VertexStream::Tangent), 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            // Formats come from the asset, because the vertex buffers are its
            // own bytes bound directly: glTF allows JOINTS_0 as unsigned byte or
            // short and WEIGHTS_0 as float or normalised integer, and this
            // single PSO has to declare whichever one this asset used.
            { "Joints",   0, m_sceneGraph.JointIndexFormat(),  static_cast<UINT>(VertexStream::Joints0),  0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            { "Weights",  0, m_sceneGraph.JointWeightFormat(), static_cast<UINT>(VertexStream::Weights0), 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
            // The lightmap UV set. Declared unconditionally, like every other
            // optional attribute: a primitive without one binds nothing in
            // that slot and the shader reads zero, which the material's
            // hasLightmapUv flag tells it to ignore.
            { "TexCoord", 1, DXGI_FORMAT_R32G32_FLOAT,    static_cast<UINT>(VertexStream::TexCoord1), 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }};

        D3D12_INPUT_LAYOUT_DESC inputLayoutDesc = {};
        inputLayoutDesc.NumElements = static_cast<UINT>(kVertexStreamCount);
        inputLayoutDesc.pInputElementDescs = inputElementDesc;

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};

        // Create root signature.
        {
            D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags
                = D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS | D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT
                | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS | D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;

            // Same as CommonStates::StaticLinearClamp
            const CD3DX12_STATIC_SAMPLER_DESC sampler(0, // register
                D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                0.f,
                16,
                D3D12_COMPARISON_FUNC_LESS_EQUAL,
                D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE,
                0.f,
                D3D12_FLOAT32_MAX,
                D3D12_SHADER_VISIBILITY_PIXEL);

            // Root signature 1.1 rather than 1.0, because
            // Descriptors::RootSignature takes a D3D12_ROOT_SIGNATURE_DESC1.
            //
            // DESCRIPTORS_VOLATILE keeps 1.0 semantics: the descriptors a table
            // points at may change between Set and Execute. The material table
            // is now written once at load and never touched again, so the
            // guarantee is not needed for it -- but the sampler table comes
            // from DirectXTK12's own heap, and promising less than VOLATILE
            // about a heap this code does not own buys nothing.
            // Must stay in lockstep with NoTextureRootSignature in
            // shaders/no_texture.fx -- the PSO is created with this signature
            // while the shader carries its own copy.
            //
            //   b0  per-node transform constants
            //   t0..t4  material textures (base colour, metallic-roughness,
            //           normal, occlusion, emissive)
            //   s0  sampler
            //   b1  per-primitive material constants
            //   b2  per-node joint matrices (skinning)
            //   t5  the lightmap atlas, bound once per pass rather than per
            //       material: every primitive samples the same one, only at a
            //       different place in it
            CD3DX12_ROOT_PARAMETER1 rootParameters[6] = {};

            rootParameters[0].InitAsConstantBufferView(0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
            const CD3DX12_DESCRIPTOR_RANGE1 materialTextureRange(
                D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                static_cast<UINT>(kMaterialTextureSlotCount), 0, 0,
                D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
            const CD3DX12_DESCRIPTOR_RANGE1 texture1SamplerRange(D3D12_DESCRIPTOR_RANGE_TYPE_SAMPLER, 1, 0, 0,
                D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
            rootParameters[1].InitAsDescriptorTable(1, &materialTextureRange, D3D12_SHADER_VISIBILITY_PIXEL);
            rootParameters[2].InitAsDescriptorTable(1, &texture1SamplerRange, D3D12_SHADER_VISIBILITY_PIXEL);
            // Visibility ALL, matching the HLSL "CBV(b1)" which defaults to ALL;
            // a narrower visibility here would not match the shader's copy.
            rootParameters[3].InitAsConstantBufferView(1, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
            // Vertex-only: skinning happens before rasterisation, so the pixel
            // shader never needs the joint matrices. Narrowing it here is safe
            // because the shader's own copy says VERTEX too.
            rootParameters[4].InitAsConstantBufferView(2, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_VERTEX);
            const CD3DX12_DESCRIPTOR_RANGE1 lightmapRange(
                D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 5, 0,
                D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
            rootParameters[5].InitAsDescriptorTable(1, &lightmapRange,
                                                    D3D12_SHADER_VISIBILITY_PIXEL);

            // Linear and clamped. A lightmap is addressed by the surface, so
            // there is nothing to tile, and a wrapping tap at a chart's edge
            // would reach into the next chart.
            const CD3DX12_STATIC_SAMPLER_DESC lightmapSampler(
                4, D3D12_FILTER_MIN_MAG_MIP_LINEAR,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f, 16,
                D3D12_COMPARISON_FUNC_NEVER,
                D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE, 0.0f, D3D12_FLOAT32_MAX,
                D3D12_SHADER_VISIBILITY_PIXEL);

            D3D12_ROOT_SIGNATURE_DESC1 rsigDesc = {};
            rsigDesc.NumParameters = static_cast<UINT>(std::size(rootParameters));
            rsigDesc.pParameters = rootParameters;
            // Must match the StaticSampler the shader's own copy of this
            // signature declares -- the PSO is built from this one while the
            // shader carries that one, and a mismatch is a creation failure.
            rsigDesc.NumStaticSamplers = 1;
            rsigDesc.pStaticSamplers = &lightmapSampler;
            rsigDesc.Flags = rootSignatureFlags;

            // Serializes and creates the ID3D12RootSignature, and records the
            // descriptor-table bit masks and per-table descriptor counts.
            m_rootSignature.SetRootSignatureDesc(rsigDesc, D3D_ROOT_SIGNATURE_VERSION_1_1);
            psoDesc.pRootSignature = m_rootSignature.GetRootSignature().Get();
        }


        const D3D12_SHADER_BYTECODE vertexShader = { g_gbuffer_vertex_shader_bytecode, sizeof(g_gbuffer_vertex_shader_bytecode) };
        psoDesc.VS = vertexShader;
        const D3D12_SHADER_BYTECODE pixelShader = { g_gbuffer_pixel_shader_bytecode, sizeof(g_gbuffer_pixel_shader_bytecode) };
        psoDesc.PS = pixelShader;
        psoDesc.InputLayout = inputLayoutDesc;
        psoDesc.BlendState = CommonStates::Opaque;
        psoDesc.DepthStencilState = CommonStates::DepthDefault;
        // glTF defines a front face as counter-clockwise. CommonStates::CullClockwise
        // arrives at the right CULLING by a double negative: measured, it is
        // CullMode = FRONT with FrontCounterClockwise = FALSE, which declares
        // clockwise triangles to be the front ones and then culls them -- so
        // counter-clockwise geometry survives, which is correct.
        //
        // What it also does is make SV_IsFrontFace true for the CLOCKWISE
        // triangles, that is, for glTF's BACK faces. PSGBuffer's double-sided
        // normal flip reads that register and would flip exactly the wrong half
        // (verified: a back-facing double-sided quad reported +Z instead of -Z).
        // Spelling the same cull the glTF way leaves which triangles survive
        // untouched and makes SV_IsFrontFace mean what the shader assumes.
        D3D12_RASTERIZER_DESC rasterizer = CommonStates::CullClockwise;
        rasterizer.CullMode = D3D12_CULL_MODE_BACK;
        rasterizer.FrontCounterClockwise = TRUE;
        psoDesc.RasterizerState = rasterizer;
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.IBStripCutValue = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;

        psoDesc.SampleMask = UINT_MAX;

        // Four targets now, not the back buffer: this pass writes the G-buffer
        // and the lighting pass resolves it. The formats have to match the
        // targets the render graph binds exactly -- a mismatch is a validation
        // error, which is why both sides read them from GBuffer.hpp.
        psoDesc.NumRenderTargets = static_cast<UINT>(kGBufferTargetCount);
        for (std::size_t target = 0; target < kGBufferTargetCount; ++target) {
            psoDesc.RTVFormats[target] =
                GBufferFormat(static_cast<GBufferTarget>(target));
        }
        // The DSV format, not the resource's: the depth buffer is created
        // typeless so the lighting pass can also read it through an SRV.
        psoDesc.DSVFormat = deviceResources->GetDepthBufferFormat();
        psoDesc.SampleDesc.Count = 1;
        psoDesc.NodeMask = 0;


        DX::ThrowIfFailed(device->CreateGraphicsPipelineState(&psoDesc, IID_GRAPHICS_PPV_ARGS(&m_pso)));
        m_pso->SetName(L"GLTF geometry (single sided)");

        // The same pipeline with culling off, for materials glTF marks
        // doubleSided -- foliage cards, cloth, leaves, anything modelled as a
        // single sheet. Without it exactly half of every such surface is
        // missing, which reads as holes rather than as a culling bug.
        //
        // Derived from the single-sided rasterizer state rather than written
        // out again, so FrontCounterClockwise is guaranteed identical: PSGBuffer
        // flips its normal on !SV_IsFrontFace, and that is only the back face if
        // both pipelines agree which winding is front.
        D3D12_RASTERIZER_DESC doubleSidedRasterizer = psoDesc.RasterizerState;
        doubleSidedRasterizer.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.RasterizerState = doubleSidedRasterizer;
        DX::ThrowIfFailed(device->CreateGraphicsPipelineState(
            &psoDesc, IID_GRAPHICS_PPV_ARGS(&m_psoDoubleSided)));
        m_psoDoubleSided->SetName(L"GLTF geometry (double sided)");

        // Depth only, for the shadow cascades. Same root signature, same input
        // layout, same vertex path -- only the target and the pixel shader
        // differ, which is the whole reason VSShadow can just be VSGBuffer.
        psoDesc.VS = {g_shadow_vertex_shader_bytecode,
                      sizeof(g_shadow_vertex_shader_bytecode)};
        psoDesc.PS = {g_shadow_pixel_shader_bytecode,
                      sizeof(g_shadow_pixel_shader_bytecode)};
        psoDesc.NumRenderTargets = 0;
        for (std::size_t target = 0; target < kGBufferTargetCount; ++target) {
            psoDesc.RTVFormats[target] = DXGI_FORMAT_UNKNOWN;
        }
        psoDesc.DSVFormat = kShadowDepthFormat;
        // Culling stays off, as set above for the double-sided pass: a caster
        // casts from either side, and clipping foliage needs both faces.
        //
        // A slope-scaled depth bias handles the acne a flat bias cannot: the
        // error from one shadow texel grows with how obliquely the light hits
        // the surface, so the correction has to as well. The constant part is
        // left at zero because PSLighting offsets along the normal instead,
        // which does not detach a shadow from the foot of what cast it.
        psoDesc.RasterizerState.DepthBias = 0;
        psoDesc.RasterizerState.SlopeScaledDepthBias = 2.0f;
        psoDesc.RasterizerState.DepthBiasClamp = 0.0f;
        DX::ThrowIfFailed(device->CreateGraphicsPipelineState(
            &psoDesc, IID_GRAPHICS_PPV_ARGS(&m_shadowPso)));
        m_shadowPso->SetName(L"GLTF shadow depth");

        // --- the lightmap bake ------------------------------------------
        // Its own root signature: the bake samples no material textures, only
        // the sun depth map, so it has nothing in common with the signature
        // above beyond the per-node transforms at b0.
        {
            CD3DX12_ROOT_PARAMETER1 bakeParameters[3] = {};
            bakeParameters[0].InitAsConstantBufferView(
                0, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
            bakeParameters[1].InitAsConstantBufferView(
                4, 0, D3D12_ROOT_DESCRIPTOR_FLAG_NONE, D3D12_SHADER_VISIBILITY_ALL);
            const CD3DX12_DESCRIPTOR_RANGE1 sunDepthRange(
                D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 13, 0,
                D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE);
            bakeParameters[2].InitAsDescriptorTable(1, &sunDepthRange,
                                                    D3D12_SHADER_VISIBILITY_PIXEL);

            const CD3DX12_STATIC_SAMPLER_DESC bakeSampler(
                3, D3D12_FILTER_COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP, D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
                D3D12_TEXTURE_ADDRESS_MODE_CLAMP, 0.0f, 16,
                D3D12_COMPARISON_FUNC_LESS_EQUAL,
                D3D12_STATIC_BORDER_COLOR_OPAQUE_WHITE, 0.0f, D3D12_FLOAT32_MAX,
                D3D12_SHADER_VISIBILITY_PIXEL);

            D3D12_ROOT_SIGNATURE_DESC1 bakeDesc = {};
            bakeDesc.NumParameters = static_cast<UINT>(std::size(bakeParameters));
            bakeDesc.pParameters = bakeParameters;
            bakeDesc.NumStaticSamplers = 1;
            bakeDesc.pStaticSamplers = &bakeSampler;
            bakeDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
                             D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
                             D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS |
                             D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS;
            m_lightmapBakeRootSignature.SetRootSignatureDesc(
                bakeDesc, D3D_ROOT_SIGNATURE_VERSION_1_1);

            D3D12_GRAPHICS_PIPELINE_STATE_DESC bakePso = {};
            bakePso.InputLayout = {inputElementDesc,
                                   static_cast<UINT>(std::size(inputElementDesc))};
            bakePso.pRootSignature =
                m_lightmapBakeRootSignature.GetRootSignature().Get();
            bakePso.VS = {g_lightmap_bake_vertex_shader_bytecode,
                          sizeof(g_lightmap_bake_vertex_shader_bytecode)};
            bakePso.PS = {g_lightmap_bake_pixel_shader_bytecode,
                          sizeof(g_lightmap_bake_pixel_shader_bytecode)};
            bakePso.BlendState = CommonStates::Opaque;
            // Culling off: the chart must be covered whichever way the triangle
            // faces, and a lightmap has no notion of a front side.
            bakePso.RasterizerState = CommonStates::CullNone;
            // No depth at all. Two triangles overlapping in UV would be a
            // broken unwrap, not something to resolve with a depth test.
            bakePso.DepthStencilState = CommonStates::DepthNone;
            bakePso.DSVFormat = DXGI_FORMAT_UNKNOWN;
            bakePso.SampleMask = UINT_MAX;
            bakePso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            bakePso.NumRenderTargets = 1;
            bakePso.RTVFormats[0] = kLightmapFormat;
            bakePso.SampleDesc.Count = 1;
            DX::ThrowIfFailed(device->CreateGraphicsPipelineState(
                &bakePso, IID_GRAPHICS_PPV_ARGS(&m_lightmapBakePso)));
            m_lightmapBakePso->SetName(L"GLTF lightmap bake");

            // One shader-visible slot, for the sun depth map.
            D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
            heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            heapDesc.NumDescriptors = 1;
            heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            DX::ThrowIfFailed(device->CreateDescriptorHeap(
                &heapDesc,
                IID_PPV_ARGS(m_lightmapBakeHeap.ReleaseAndGetAddressOf())));
            m_lightmapBakeHeap->SetName(L"GLTF lightmap bake heap");
            m_lightmapBakeDescriptorSize = device->GetDescriptorHandleIncrementSize(
                D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        // Called from here rather than exposed as another step the caller has
        // to remember: it needs the device and the built scene graph, which is
        // exactly what being called at the end of PreparePSO guarantees.
        BuildMaterialDescriptorTables(deviceResources);
        BuildLightmapCharts();
    }

    void GLTFAdapter::BuildLightmapCharts() {
        // World transforms first: a chart's size comes from how much world
        // surface the instance covers, which is a property of where it was
        // placed and how it was scaled, not of the mesh. Done here rather than
        // assumed, so this does not depend on the caller having framed the
        // scene yet.
        m_sceneGraph.UpdateTransforms(ScenePlacement());

        m_lightmapCharts = 0;
        m_lightmapMissingUv = 0;
        m_lightmapArea = 0.0f;

        // One entry per instance, in draw order, so a chart can be written
        // back to the primitive it was sized for.
        std::vector<float> areas;
        std::vector<PrimitiveResource *> primitives;
        for (SceneNode &node : m_sceneGraph.NodesForResourceBinding()) {
            for (PrimitiveResource &primitive : node.primitives) {
                primitive.material.lightmapScaleOffset =
                    DirectX::XMFLOAT4(0.0f, 0.0f, 0.0f, 0.0f);

                // Only static geometry can be baked, and only geometry with a
                // non-overlapping UV set can be addressed.
                if (node.movesAtRuntime) {
                    continue;
                }
                if (!primitive.hasLightmapUv) {
                    ++m_lightmapMissingUv;
                    continue;
                }

                DirectX::SimpleMath::Vector3 centre;
                DirectX::SimpleMath::Vector3 extent;
                if (!PrimitiveWorldBounds(primitive, node.worldTransform, centre,
                                          extent)) {
                    continue;
                }
                // The face a shadow falls on: the product of the two largest
                // full extents. A primitive is usually slab-shaped, so this is
                // much closer than the box's whole surface area, which would
                // over-count a flat panel by a factor of two.
                float edges[3] = {2.0f * extent.x, 2.0f * extent.y, 2.0f * extent.z};
                std::sort(std::begin(edges), std::end(edges));
                const float area = edges[1] * edges[2];
                if (!(area > 0.0f)) {
                    continue;
                }
                areas.push_back(area);
                primitives.push_back(&primitive);
            }
        }

        m_lightmap = PackLightmap(areas, m_lightmapSetup);
        for (std::size_t i = 0; i < primitives.size(); ++i) {
            const LightmapChart &chart = m_lightmap.charts[i];
            if (!chart.placed) {
                continue;
            }
            primitives[i]->material.lightmapScaleOffset = chart.scaleOffset;
            ++m_lightmapCharts;
            m_lightmapArea += areas[i];
        }

        std::fprintf(stderr,
                     "lightmap: %zu chart(s) of %zu static instance(s), "
                     "%.0f m2 at %.0f texels/m, %u x %u atlas (%.1f%% used)%s\n",
                     m_lightmapCharts, areas.size(), m_lightmapArea,
                     static_cast<double>(m_lightmapSetup.texelsPerMetre),
                     m_lightmap.atlasResolution, m_lightmap.atlasResolution,
                     m_lightmap.atlasResolution == 0
                         ? 0.0
                         : 100.0 * static_cast<double>(m_lightmap.texelsPlaced) /
                               (static_cast<double>(m_lightmap.atlasResolution) *
                                m_lightmap.atlasResolution),
                     m_lightmap.complete ? "" : " -- DID NOT FIT");
        if (m_lightmapMissingUv != 0) {
            // The number an unwrapper exists to fix. Said plainly because a
            // scene that bakes nothing looks exactly like a scene whose bake
            // is broken.
            std::fprintf(stderr,
                         "lightmap: %zu static instance(s) have no TEXCOORD_1 and "
                         "cannot be baked\n",
                         m_lightmapMissingUv);
        }
        std::fflush(stderr);
    }

    void GLTFAdapter::UpdateSceneTransforms() {
        // The scene placement, unchanged: this is the root transform that every
        // node's own transform is composed onto, rather than the single matrix
        // every mesh was drawn with.
        // Via ScenePlacement so retargeting, which has to aim through this
        // same transform, cannot drift from what is actually drawn.
        //
        // Top-down pass: each node's world transform is its local transform
        // composed with its parent's, and hidden subtrees drop out of the draw
        // order here rather than being tested per draw.
        m_sceneGraph.UpdateTransforms(ScenePlacement());
    }

    void GLTFAdapter::SetLightmapAtlasSrv(D3D12_CPU_DESCRIPTOR_HANDLE srv) {
        // Copied every frame rather than once: the graph may recreate the
        // atlas, and one CopyDescriptorsSimple of a single descriptor is far
        // cheaper than tracking whether it did.
        if (!m_materialTableHeap || srv.ptr == 0) {
            return;
        }
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        if (FAILED(m_materialTableHeap->GetDevice(IID_PPV_ARGS(device.GetAddressOf())))) {
            return;
        }
        CD3DX12_CPU_DESCRIPTOR_HANDLE destination(
            m_materialTableHeap->GetCPUDescriptorHandleForHeapStart(),
            static_cast<INT>(m_lightmapAtlasSlot), m_materialDescriptorIncrement);
        device->CopyDescriptorsSimple(1, destination, srv,
                                      D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        m_lightmapAtlasBound = true;
    }

    void GLTFAdapter::RecordGeometry(ID3D12GraphicsCommandList *commandList,
                                     const DirectX::SimpleMath::Matrix &viewMatrix,
                                     const DirectX::SimpleMath::Matrix &projectionMatrix) {
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        DX::ThrowIfFailed(commandList->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));
        // https://www.reddit.com/r/GraphicsProgramming/comments/1bnb08z/dx12_confusion_about_root_signatures/
        // https://stackoverflow.com/questions/38535725/what-is-the-point-of-d3d12s-setgraphicsrootsignature
        // The "root signature" in DirectX 12 provides the common layout information for sharing data between the CPU data structures and the GPU shader language execution.
        // NOTE: Although PSO state explicitly declare root signature and primitive topology, we still need to assign it by ourselves because each pso doesn't share this information
        // The setting of pRootSignature through Set*RootSignature() is like holding CPU and GPU function signature along with their constant buffer, texture binding so that when swithcing PSO or partial resources, the other binding doesn't change
        // While the creation of PSO with pRootSignature is like providing hardware and driver info to optimize shader compilation
        // https://learn.microsoft.com/en-us/windows/win32/direct3d12/root-signatures-overview
        // The render targets, the depth target, the viewport, the scissor rect
        // and the clears that used to be here are the render graph's: it has
        // already bound the G-buffer and cleared it before calling this.

        // Descriptor heap bindings do not survive a command list reset, and
        // Application's DirectXTK12 draws rebind their own heaps later in the
        // frame, so re-establish ours here. The sampler heap is registered once;
        // the dynamic SRV heap registers itself as it commits, and the binder
        // reissues SetDescriptorHeaps with both whenever either changes.
        m_heapBinder.Reset(commandList);
        m_heapBinder.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, m_commonStates->Heap());
        m_heapBinder.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                                       m_materialTableHeap.Get());

        commandList->SetGraphicsRootSignature(m_rootSignature.GetRootSignature().Get());
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

        // Once per pass, not once per draw. Nothing switches descriptor heaps
        // mid-pass any more -- the material heap is built at load and bound
        // above -- so a root table set here stays valid for every draw after
        // it. It used to have to come after each draw's descriptor commit,
        // because a commit that outgrew its heap invalidated every bound table.
        commandList->SetGraphicsRootDescriptorTable(2, m_commonStates->AnisotropicWrap());
        // The lightmap atlas, also once per pass: every primitive samples the
        // same texture, only at a different place in it.
        if (m_lightmapAtlasBound) {
            CD3DX12_GPU_DESCRIPTOR_HANDLE lightmap(
                m_materialTableHeap->GetGPUDescriptorHandleForHeapStart(),
                static_cast<INT>(m_lightmapAtlasSlot), m_materialDescriptorIncrement);
            commandList->SetGraphicsRootDescriptorTable(5, lightmap);
        }

        // Selected per primitive below, tracked so a run of primitives sharing
        // a facing does not re-set it.
        ID3D12PipelineState *boundPso = nullptr;
        // The material block bound by the previous draw. Consecutive primitives
        // very often share a material, and this is now an integer compare
        // against a block index rather than a comparison of five descriptor
        // handles followed by a staging decision.
        std::optional<std::uint32_t> boundMaterialTable;

        // Carried across the reset, since the shadow passes already counted
        // into them this frame. See m_shadowStatsStale.
        const std::size_t shadowDraws = m_shadowStatsStale ? 0 : m_stats.shadowDraws;
        const std::size_t shadowCulled = m_shadowStatsStale ? 0 : m_stats.shadowCulled;
        m_stats = GeometryStats{};
        m_stats.shadowDraws = shadowDraws;
        m_stats.shadowCulled = shadowCulled;
        m_shadowStatsStale = true;

        // The frustum, in world space, from the same matrices the vertex
        // shader will use. Built once per frame, not per primitive.
        const ViewFrustum frustum =
            ViewFrustum::FromViewProjection(viewMatrix * projectionMatrix);

        // Lights are a property of the scene, not of a node, so they are packed
        // once and copied into every node's constant buffer.
        //
        // Nothing in the G-buffer pass reads them any more -- the deferred
        // lighting pass has its own, wider, light buffer. They stay filled
        // because PBR_Constants' layout is pinned by static_asserts either way,
        // and because a forward pass added later will want them.
        //
        // UpdateSceneTransforms must have placed them before this runs.
        ShaderLight frameLights[kMaxShaderLights] = {};
        const std::size_t lightCount =
            m_sceneGraph.GatherShaderLights(frameLights, kMaxShaderLights);

        // b2 has to be bound for every draw, skinned or not: the vertex shader
        // declares the cbuffer unconditionally, and reading an unbound root CBV
        // is undefined. Unskinned nodes share this one all-identity allocation
        // rather than each paying for 8 KB of identities.
        SharedGraphicsResource identityJoints =
            GraphicsMemory::Get(device.Get()).AllocateConstant<JointMatrixConstants>();
        {
            auto *joints = static_cast<JointMatrixConstants *>(identityJoints.Memory());
            for (std::size_t j = 0; j < kMaxJointMatrices; ++j) {
                joints->joints[j] = XMMatrixIdentity();
            }
        }
        std::vector<DirectX::SimpleMath::Matrix> jointMatrices;
        // One constant buffer per node, because each now carries its own world
        // matrix.
        //
        // NOTE: nothing in this project calls GraphicsMemory::Commit(), which is
        // what retires allocator pages against a fence. Measured over 45 s the
        // working set is stable, so this is not a leak -- pages are being
        // recycled -- but recycling without the fence is what Commit() exists to
        // prevent, so a page could in principle be reused while the GPU is still
        // reading it. Pre-existing and never observed to misbehave; flagged
        // rather than changed.
        for (const std::size_t nodeIndex : m_sceneGraph.DrawOrder()) {
            const SceneNode &node = m_sceneGraph.Nodes()[nodeIndex];

            PIXBeginEvent(commandList, PIX_COLOR_DEFAULT, L"Begin GLTF node");

            PBREffectConstants pbrEffectConstant = {};
            pbrEffectConstant.eyePosition = m_camera->mEye;

            // World and world-inverse-transpose, in the byte order HLSL's
            // column-major cbuffer read expects. Both were wrong before:
            // `world` went in untransposed, and worldInverseTranspose was never
            // written at all, so VSOutput::WorldPosition was garbage and
            // WorldNormal was zero. Nothing consumed either -- the pixel shader
            // returned base colour -- but the lights below are in world space,
            // so the shading about to be written needs them both.
            PackWorldMatrices(node.worldTransform, pbrEffectConstant.world,
                              pbrEffectConstant.worldInverseTranspose);

            pbrEffectConstant.worldViewProj = XMMatrixTranspose(
                XMMatrixMultiply(XMMatrixMultiply(node.worldTransform, viewMatrix), projectionMatrix));

            std::memcpy(pbrEffectConstant.lights, frameLights, sizeof(frameLights));
            pbrEffectConstant.lightCount = static_cast<int>(lightCount);
            // NOTE: Set*Root* only set BufferLocation instead of SizeInBytes, which may cause GPU to crash if shader access data out of bound
            // Therefore, it's suitable for frequently changing resource.
            // https://gamedev.net/forums/topic/678623-d3d12-using-setgraphicsrootview-functions/#post-5291945
            SharedGraphicsResource cBufferResource =
                GraphicsMemory::Get(device.Get()).AllocateConstant(pbrEffectConstant);
            commandList->SetGraphicsRootConstantBufferView(0, cBufferResource.GpuAddress());

            // Joint matrices are per (skin, node): the formula divides out this
            // node's world transform, so two nodes sharing a skin do not share
            // matrices. Computed here rather than in UpdateTransforms so an
            // unskinned scene pays nothing.
            D3D12_GPU_VIRTUAL_ADDRESS jointsAddress = identityJoints.GpuAddress();
            SharedGraphicsResource nodeJoints;
            if (node.IsSkinned() && m_sceneGraph.ComputeJointMatrices(nodeIndex, jointMatrices)) {
                nodeJoints =
                    GraphicsMemory::Get(device.Get()).AllocateConstant<JointMatrixConstants>();
                auto *joints = static_cast<JointMatrixConstants *>(nodeJoints.Memory());
                const std::size_t count = std::min(jointMatrices.size(), kMaxJointMatrices);
                for (std::size_t j = 0; j < count; ++j) {
                    // Transposed for the same reason PBR_World is: HLSL reads
                    // cbuffer matrices column-major.
                    joints->joints[j] = XMMatrixTranspose(jointMatrices[j]);
                }
                // A skin larger than the array is reported by SceneGraph; the
                // remainder stays identity so an out-of-range index is harmless.
                for (std::size_t j = count; j < kMaxJointMatrices; ++j) {
                    joints->joints[j] = XMMatrixIdentity();
                }
                jointsAddress = nodeJoints.GpuAddress();
            }
            commandList->SetGraphicsRootConstantBufferView(4, jointsAddress);
            ++m_stats.nodes;
            if (node.IsSkinned()) {
                ++m_stats.skinnedNodes;
            }

            for (const PrimitiveResource &primitive : node.primitives) {
                // Frustum culling, before anything is bound or allocated for
                // this primitive: the point is to skip the constant buffer, the
                // descriptor staging and the draw, not just the draw.
                //
                // A skinned primitive is never culled. Its vertices are moved
                // by the joint matrices, so the authored POSITION bounds do not
                // contain the deformed mesh -- the retargeted character's hips
                // are relocated outright. Bounding the posed mesh would mean
                // transforming the bounds by every joint, which is not worth it
                // for the handful of skinned primitives in any of these scenes.
                if (m_frustumCulling && !primitive.material.isSkinned) {
                    DirectX::SimpleMath::Vector3 centre;
                    DirectX::SimpleMath::Vector3 extent;
                    if (PrimitiveWorldBounds(primitive, node.worldTransform, centre,
                                             extent)) {
                        if (!frustum.Intersects(centre, extent)) {
                            ++m_stats.culledPrimitives;
                            continue;
                        }
                    } else {
                        // No authored bounds, so no opinion: drawn.
                        ++m_stats.unculledPrimitives;
                    }
                } else {
                    ++m_stats.unculledPrimitives;
                }

                // Culling comes from the material, so the pipeline is chosen
                // here rather than once for the frame.
                ID3D12PipelineState *const wantedPso = primitive.material.doubleSided
                                                           ? m_psoDoubleSided.Get()
                                                           : m_pso.Get();
                if (wantedPso != boundPso) {
                    commandList->SetPipelineState(wantedPso);
                    boundPso = wantedPso;
                    ++m_stats.pipelineSwitches;
                }

                // Per-primitive material factors at b1.
                SharedGraphicsResource materialResource =
                    GraphicsMemory::Get(device.Get()).AllocateConstant(primitive.material);
                commandList->SetGraphicsRootConstantBufferView(3, materialResource.GpuAddress());

                // The material's five descriptors already sit in
                // m_materialTableHeap, so this is a root table set at a known
                // offset rather than a descriptor copy. Skipped outright when
                // the previous draw used the same block.
                if (boundMaterialTable != primitive.materialTableIndex) {
                    commandList->SetGraphicsRootDescriptorTable(
                        1, MaterialTable(primitive.materialTableIndex));
                    boundMaterialTable = primitive.materialTableIndex;
                    ++m_stats.materialTableBinds;
                }

                commandList->IASetVertexBuffers(0, static_cast<UINT>(kVertexStreamCount),
                                                primitive.vertexBufferViews);
                commandList->IASetIndexBuffer(&primitive.indexBufferView);

                commandList->DrawIndexedInstanced(primitive.indexCount, 1, 0, 0, 0);
                ++m_stats.draws;
                m_stats.triangles += primitive.indexCount / 3;
            }

            PIXEndEvent(commandList);
        }
    }

    void GLTFAdapter::BuildMaterialDescriptorTables(
        std::shared_ptr<DX::DeviceResources> deviceResources) {
        // Every material's five SRV descriptors, copied into one GPU-visible
        // heap once, here, instead of being copied into a dynamic heap on every
        // draw that changes material.
        //
        // What makes this possible is that a material's table is nothing but
        // those five descriptors, and they never change after load: the images
        // are created once by PrepareImage and the fallback once by
        // PrepareFallbackTexture. There was never anything per-frame about
        // them. The dynamic heap was copying the same bytes to the same place
        // every frame because that is what the upstream staging model does, not
        // because the data moved.
        //
        // Deduplicated on the five-slot texture tuple rather than per
        // primitive. Two materials naming the same five images are the same
        // table, and in a converted scene a great many of them are: everything
        // with no textures at all shares the block that is five copies of the
        // fallback.
        auto* device = deviceResources->GetD3DDevice();
        m_materialDescriptorIncrement = device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        // npos stands in for "this slot has no texture", so the tuple is a
        // plain comparable key.
        constexpr std::size_t kNoTexture = static_cast<std::size_t>(-1);
        using TableKey = std::array<std::size_t, kMaterialTextureSlotCount>;

        std::map<TableKey, std::uint32_t> blocks;
        std::vector<D3D12_CPU_DESCRIPTOR_HANDLE> sources;
        std::uint32_t nextBlock = 0;
        std::size_t primitiveCount = 0;

        for (SceneNode& node : m_sceneGraph.NodesForResourceBinding()) {
            for (PrimitiveResource& primitive : node.primitives) {
                ++primitiveCount;
                TableKey key = {};
                for (std::size_t slot = 0; slot < kMaterialTextureSlotCount; ++slot) {
                    key[slot] = primitive.textureImageIndex[slot].value_or(kNoTexture);
                }

                const auto existing = blocks.find(key);
                if (existing != blocks.end()) {
                    primitive.materialTableIndex = existing->second;
                    continue;
                }

                // Every slot of the table gets a descriptor: the material's
                // texture where it has one, the white fallback otherwise.
                // Leaving a slot unwritten would put an uninitialised
                // descriptor in a bound table, which is invalid whether the
                // shader reads it or not.
                for (std::size_t slot = 0; slot < kMaterialTextureSlotCount; ++slot) {
                    sources.push_back(
                        key[slot] != kNoTexture
                            ? m_imageDescriptors[key[slot]].GetDescriptorHandle()
                            : m_fallbackTextureDescriptor.GetDescriptorHandle());
                }
                blocks.emplace(key, nextBlock);
                primitive.materialTableIndex = nextBlock;
                ++nextBlock;
            }
        }

        m_materialTableCount = blocks.size();
        // One slot past the material blocks, for the lightmap atlas. It lives
        // in this heap because the G-buffer pass has to bind a shader-visible
        // heap anyway, and the atlas is the one SRV that is the same for every
        // draw -- so it is bound once per pass rather than per material.
        m_lightmapAtlasSlot = sources.size();
        if (sources.empty()) {
            // A scene with no primitives at all. Nothing to bind, and
            // RecordGeometry has nothing to draw either.
            m_materialTableHeap.Reset();
            m_materialTableStart = {};
            return;
        }

        D3D12_DESCRIPTOR_HEAP_DESC heapDesc = {};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.NumDescriptors = static_cast<UINT>(sources.size() + 1);
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        DX::ThrowIfFailed(device->CreateDescriptorHeap(
            &heapDesc, IID_PPV_ARGS(m_materialTableHeap.ReleaseAndGetAddressOf())));
        m_materialTableHeap->SetName(L"GLTFAdapter material tables");
        m_materialTableStart = m_materialTableHeap->GetGPUDescriptorHandleForHeapStart();

        // One destination range -- the whole heap is contiguous and filled in
        // block order -- against one source range per descriptor. Chunked
        // because the number of source ranges is scene-sized and there is no
        // reason to hand a driver an unbounded array in a single call.
        constexpr UINT kMaxRangesPerCall = 1024;
        const CD3DX12_CPU_DESCRIPTOR_HANDLE destStart(
            m_materialTableHeap->GetCPUDescriptorHandleForHeapStart());
        for (std::size_t offset = 0; offset < sources.size();
             offset += kMaxRangesPerCall) {
            const UINT count = static_cast<UINT>(
                std::min<std::size_t>(kMaxRangesPerCall, sources.size() - offset));
            const CD3DX12_CPU_DESCRIPTOR_HANDLE destination(
                destStart, static_cast<INT>(offset), m_materialDescriptorIncrement);
            const D3D12_CPU_DESCRIPTOR_HANDLE destinations[] = {destination};
            const UINT destinationSizes[] = {count};
            device->CopyDescriptors(1, destinations, destinationSizes, count,
                                    sources.data() + offset, nullptr,
                                    D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        }

        // stderr, like the other load-time lines: stdout is redirected to a
        // console this process creates, so it is not what a pipe captures.
        std::fprintf(stderr,
                     "material tables: %zu distinct of %zu primitives, "
                     "%zu descriptors (%.1f KB)\n",
                     m_materialTableCount, primitiveCount, sources.size(),
                     static_cast<double>(sources.size() *
                                         m_materialDescriptorIncrement) /
                         1024.0);
        std::fflush(stderr);
    }

    void GLTFAdapter::RecordLightmapBake(
        ID3D12GraphicsCommandList *commandList,
        const DirectX::SimpleMath::Matrix &sunViewProjection,
        float sunTexelWorldSize, D3D12_CPU_DESCRIPTOR_HANDLE sunDepth) {
        if (!m_lightmapBakePso || m_lightmapCharts == 0) {
            return;
        }
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        DX::ThrowIfFailed(commandList->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));

        // The sun depth map, copied into the bake's own one-slot heap. The
        // material heap is not bound here at all: the bake samples no material
        // textures.
        device->CopyDescriptorsSimple(
            1, m_lightmapBakeHeap->GetCPUDescriptorHandleForHeapStart(), sunDepth,
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        ID3D12DescriptorHeap *heaps[] = {m_lightmapBakeHeap.Get()};
        commandList->SetDescriptorHeaps(1, heaps);
        commandList->SetGraphicsRootSignature(
            m_lightmapBakeRootSignature.GetRootSignature().Get());
        commandList->SetPipelineState(m_lightmapBakePso.Get());
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        commandList->SetGraphicsRootDescriptorTable(
            2, m_lightmapBakeHeap->GetGPUDescriptorHandleForHeapStart());

        const float inverseResolution =
            1.0f / static_cast<float>(kLightmapSunDepthResolution);

        // The static half only: a moving primitive has no chart, and baking
        // where it happens to be standing would leave its shadow behind when
        // it moved.
        for (const std::size_t nodeIndex : m_sceneGraph.StaticDrawOrder()) {
            const SceneNode &node = m_sceneGraph.Nodes()[nodeIndex];

            PBREffectConstants pbrEffectConstant = {};
            PackWorldMatrices(node.worldTransform, pbrEffectConstant.world,
                              pbrEffectConstant.worldInverseTranspose);
            // worldViewProj is unused by the bake -- the vertex shader
            // rasterises into the chart, not into a camera -- but the struct
            // is shared, so it is filled rather than left as rubbish.
            pbrEffectConstant.worldViewProj =
                XMMatrixTranspose(XMMatrixMultiply(node.worldTransform,
                                                   sunViewProjection));
            SharedGraphicsResource nodeConstants =
                GraphicsMemory::Get(device.Get()).AllocateConstant(pbrEffectConstant);
            commandList->SetGraphicsRootConstantBufferView(0,
                                                           nodeConstants.GpuAddress());

            for (const PrimitiveResource &primitive : node.primitives) {
                // No chart: either the mesh has no lightmap UVs or the pack
                // could not place it. Either way there is nowhere to write.
                if (primitive.material.lightmapScaleOffset.x <= 0.0f) {
                    continue;
                }

                LightmapBakeConstants bake = {};
                bake.sunViewProjection = XMMatrixTranspose(sunViewProjection);
                bake.chartScaleOffset = primitive.material.lightmapScaleOffset;
                bake.depthBias = m_lightmapDepthBias;
                bake.normalBias = m_lightmapNormalBias;
                bake.sunTexelSize = sunTexelWorldSize;
                bake.sunUvTexel = inverseResolution;
                SharedGraphicsResource bakeConstants =
                    GraphicsMemory::Get(device.Get()).AllocateConstant(bake);
                commandList->SetGraphicsRootConstantBufferView(
                    1, bakeConstants.GpuAddress());

                commandList->IASetVertexBuffers(
                    0, static_cast<UINT>(kVertexStreamCount),
                    primitive.vertexBufferViews);
                commandList->IASetIndexBuffer(&primitive.indexBufferView);
                commandList->DrawIndexedInstanced(primitive.indexCount, 1, 0, 0, 0);
                ++m_lightmapBakedPrimitives;
            }
        }
    }

    void GLTFAdapter::RecordShadowGeometry(
        ID3D12GraphicsCommandList *commandList,
        const DirectX::SimpleMath::Matrix &lightViewProjection, bool movingOnly) {
        Microsoft::WRL::ComPtr<ID3D12Device> device;
        DX::ThrowIfFailed(commandList->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));

        // Descriptor heap bindings do not survive a command list reset, and the
        // same binder serves every pass this class records.
        m_heapBinder.Reset(commandList);
        m_heapBinder.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
                                       m_commonStates->Heap());
        m_heapBinder.SetDescriptorHeap(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                                       m_materialTableHeap.Get());

        commandList->SetGraphicsRootSignature(m_rootSignature.GetRootSignature().Get());
        commandList->SetPipelineState(m_shadowPso.Get());
        commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        // Once per cascade rather than once per draw -- see RecordGeometry.
        commandList->SetGraphicsRootDescriptorTable(
            2, m_commonStates->AnisotropicWrap());

        // Zeroed on the frame's first shadow recording only, so the rest add
        // to it rather than replacing it. That first recording may be either
        // half -- a frame with no refitted cascade never draws the static one.
        if (m_shadowStatsStale) {
            m_stats.shadowDraws = 0;
            m_stats.shadowCulled = 0;
            m_shadowStatsStale = false;
        }

        // Culled against the cascade, not the camera: a caster outside the
        // cascade's box contributes nothing to this tile, and the cascades are
        // small enough that this rejects most of the scene.
        const ViewFrustum frustum = ViewFrustum::FromViewProjection(lightViewProjection);

        // The material block bound by the previous draw, so a run of
        // primitives sharing a material costs one root table set.
        std::optional<std::uint32_t> boundMaterialTable;

        // b2 has to be bound even though only a skinned primitive reads it,
        // for the same reason as in RecordGeometry: the vertex shader declares
        // the cbuffer unconditionally.
        SharedGraphicsResource identityJoints =
            GraphicsMemory::Get(device.Get()).AllocateConstant<JointMatrixConstants>();
        {
            auto *joints = static_cast<JointMatrixConstants *>(identityJoints.Memory());
            for (std::size_t j = 0; j < kMaxJointMatrices; ++j) {
                joints->joints[j] = XMMatrixIdentity();
            }
        }
        std::vector<DirectX::SimpleMath::Matrix> jointMatrices;

        // Only the half this call is for, so a frame that redraws nothing but
        // the movers does not walk the whole scene once per cascade to find
        // them.
        const std::vector<std::size_t> &order = movingOnly
                                                    ? m_sceneGraph.MovingDrawOrder()
                                                    : m_sceneGraph.StaticDrawOrder();
        for (const std::size_t nodeIndex : order) {
            const SceneNode &node = m_sceneGraph.Nodes()[nodeIndex];

            PBREffectConstants pbrEffectConstant = {};
            // The light's matrix where the camera's would go. PBR_World and the
            // inverse transpose are filled because PSShadow shares VSOutput
            // with the G-buffer path, not because anything reads them.
            PackWorldMatrices(node.worldTransform, pbrEffectConstant.world,
                              pbrEffectConstant.worldInverseTranspose);
            pbrEffectConstant.worldViewProj = XMMatrixTranspose(
                XMMatrixMultiply(node.worldTransform, lightViewProjection));

            SharedGraphicsResource cBufferResource =
                GraphicsMemory::Get(device.Get()).AllocateConstant(pbrEffectConstant);
            commandList->SetGraphicsRootConstantBufferView(0, cBufferResource.GpuAddress());

            D3D12_GPU_VIRTUAL_ADDRESS jointsAddress = identityJoints.GpuAddress();
            SharedGraphicsResource nodeJoints;
            if (node.IsSkinned() &&
                m_sceneGraph.ComputeJointMatrices(nodeIndex, jointMatrices)) {
                nodeJoints = GraphicsMemory::Get(device.Get())
                                 .AllocateConstant<JointMatrixConstants>();
                auto *joints = static_cast<JointMatrixConstants *>(nodeJoints.Memory());
                const std::size_t count =
                    std::min(jointMatrices.size(), kMaxJointMatrices);
                for (std::size_t j = 0; j < count; ++j) {
                    joints->joints[j] = XMMatrixTranspose(jointMatrices[j]);
                }
                for (std::size_t j = count; j < kMaxJointMatrices; ++j) {
                    joints->joints[j] = XMMatrixIdentity();
                }
                jointsAddress = nodeJoints.GpuAddress();
            }
            commandList->SetGraphicsRootConstantBufferView(4, jointsAddress);

            for (const PrimitiveResource &primitive : node.primitives) {
                if (!primitive.material.isSkinned) {
                    DirectX::SimpleMath::Vector3 centre;
                    DirectX::SimpleMath::Vector3 extent;
                    if (PrimitiveWorldBounds(primitive, node.worldTransform, centre,
                                             extent) &&
                        !frustum.Intersects(centre, extent)) {
                        ++m_stats.shadowCulled;
                        continue;
                    }
                }

                // The material constants, for the alpha clip in PSShadow. An
                // opaque primitive does not read them, but binding per
                // primitive is cheaper than branching on whether to.
                SharedGraphicsResource materialResource =
                    GraphicsMemory::Get(device.Get()).AllocateConstant(primitive.material);
                commandList->SetGraphicsRootConstantBufferView(
                    3, materialResource.GpuAddress());

                // The whole table is bound even though only the base colour is
                // sampled: a bound table with an uninitialised entry is invalid
                // whether the shader reads it or not, and the block already
                // holds all five.
                if (boundMaterialTable != primitive.materialTableIndex) {
                    commandList->SetGraphicsRootDescriptorTable(
                        1, MaterialTable(primitive.materialTableIndex));
                    boundMaterialTable = primitive.materialTableIndex;
                }

                commandList->IASetVertexBuffers(0, static_cast<UINT>(kVertexStreamCount),
                                                primitive.vertexBufferViews);
                commandList->IASetIndexBuffer(&primitive.indexBufferView);
                commandList->DrawIndexedInstanced(primitive.indexCount, 1, 0, 0, 0);
                ++m_stats.shadowDraws;
            }
        }
    }

    void GLTFAdapter::UpdateAnimation(float deltaSeconds) {
        const auto &animations = m_sceneGraph.Animations();
        if (animations.empty()) {
            return;
        }

        // Pick the first clip once, so an animated asset moves without the UI
        // being touched. After that an empty selection means the user asked for
        // the authored pose, and must be left alone.
        if (!m_animationAutoSelected) {
            m_animationAutoSelected = true;
            m_activeAnimation = 0;
            m_animationTime = 0.0f;
        }
        if (!m_activeAnimation.has_value()) {
            return;
        }
        if (m_activeAnimation.value() >= animations.size()) {
            m_activeAnimation = 0;
        }

        const SceneAnimation &animation = animations[m_activeAnimation.value()];

        if (m_animationPlaying && animation.duration > 0.0f) {
            m_animationTime += deltaSeconds * m_animationSpeed;
            if (m_animationLoop) {
                // fmod keeps a negative speed working too, where plain
                // subtraction would walk off the start of the timeline.
                m_animationTime = std::fmod(m_animationTime, animation.duration);
                if (m_animationTime < 0.0f) {
                    m_animationTime += animation.duration;
                }
            } else {
                m_animationTime = std::min(m_animationTime, animation.duration);
            }
        }

        // Poses the nodes. Render's UpdateTransforms then composes them and
        // ComputeJointMatrices turns the joints into the b2 matrices.
        m_sceneGraph.ApplyAnimation(m_activeAnimation.value(), m_animationTime);
    }

    void GLTFAdapter::ShowImgui(UiSection section) {
        switch (section) {
            case UiSection::Render:      ShowRenderUi(); break;
            case UiSection::Scene:       ShowSceneUi(); break;
            case UiSection::Performance: ShowPerformanceUi(); break;
            case UiSection::Neural:      break;
        }
    }

    void GLTFAdapter::ShowRenderUi() {
        // Here rather than beside the counts it changes, because it decides
        // what is drawn. What it saved is in the Performance tab.
        ImGui::Checkbox("Frustum culling", &m_frustumCulling);
    }

    void GLTFAdapter::ShowSceneUi() {
        ImGui::SliderFloat3("GLTF position", m_position, -10.0, 10.0);
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Animation", ImGuiTreeNodeFlags_DefaultOpen)) {
            const auto &animations = m_sceneGraph.Animations();
            if (animations.empty()) {
                ImGui::TextUnformatted("This asset has no animations.");
            } else {
                // "none" restores the authored pose, which is the only way to
                // see the bind pose once playback has started.
                const char *current = m_activeAnimation.has_value()
                                          ? animations[m_activeAnimation.value()].name.c_str()
                                          : "(none -- authored pose)";
                if (ImGui::BeginCombo("Clip", current)) {
                    if (ImGui::Selectable("(none -- authored pose)",
                                          !m_activeAnimation.has_value())) {
                        m_activeAnimation.reset();
                        m_sceneGraph.ResetToBasePose();
                    }
                    for (std::size_t i = 0; i < animations.size(); ++i) {
                        const bool selected = m_activeAnimation.has_value() &&
                                              m_activeAnimation.value() == i;
                        if (ImGui::Selectable(animations[i].name.c_str(), selected)) {
                            m_activeAnimation = i;
                            m_animationTime = 0.0f;
                        }
                    }
                    ImGui::EndCombo();
                }

                if (m_activeAnimation.has_value()) {
                    const SceneAnimation &animation = animations[m_activeAnimation.value()];
                    ImGui::Checkbox("Play", &m_animationPlaying);
                    ImGui::SameLine();
                    ImGui::Checkbox("Loop", &m_animationLoop);
                    ImGui::SliderFloat("Speed", &m_animationSpeed, -2.0f, 2.0f);
                    // Scrubbing while paused is the useful case; dragging while
                    // playing just gets overwritten on the next frame.
                    if (ImGui::SliderFloat("Time", &m_animationTime, 0.0f,
                                           animation.duration > 0.0f ? animation.duration
                                                                     : 1.0f)) {
                        m_sceneGraph.ApplyAnimation(m_activeAnimation.value(),
                                                    m_animationTime);
                    }
                    ImGui::Text("%.3f / %.3f s, %zu channels", m_animationTime,
                                animation.duration, animation.channels.size());
                    if (animation.skippedChannels != 0) {
                        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                                           "%zu channel(s) skipped (morph weights "
                                           "are not supported)",
                                           animation.skippedChannels);
                    }
                }
            }
        }
        ImGui::Separator();
        if (ImGui::CollapsingHeader("Images")) {
            const auto &spaces = m_imageColorSpaces.perImage;
            std::size_t srgbCount = 0;
            for (const ImageColorSpace space : spaces) {
                if (space == ImageColorSpace::Srgb) {
                    ++srgbCount;
                }
            }
            ImGui::Text("%zu images: %zu sRGB, %zu linear", spaces.size(), srgbCount,
                        spaces.size() - srgbCount);
            if (!m_imageColorSpaces.conflicts.empty()) {
                ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                                   "%zu image(s) used as both colour and linear data",
                                   m_imageColorSpaces.conflicts.size());
            }
            for (std::size_t i = 0; i < spaces.size(); ++i) {
                ImGui::Text("  image %zu: %s", i, ToString(spaces[i]));
            }
        }
        if (ImGui::CollapsingHeader("Scene hierarchy", ImGuiTreeNodeFlags_DefaultOpen)) {
            m_sceneGraph.DrawHierarchyUI();
        }
    }

    void GLTFAdapter::ShowPerformanceUi() {
        if (!ImGui::CollapsingHeader("Scene cost", ImGuiTreeNodeFlags_DefaultOpen)) {
            return;
        }
        ImGui::Text("%zu nodes drawn, %zu of them skinned", m_stats.nodes,
                    m_stats.skinnedNodes);
        ImGui::Text("%zu draws, %zu triangles", m_stats.draws, m_stats.triangles);
        if (m_frustumCulling) {
            const std::size_t considered = m_stats.draws + m_stats.culledPrimitives;
            ImGui::Text("%zu of %zu primitives culled (%.0f%%)",
                        m_stats.culledPrimitives, considered,
                        considered != 0 ? 100.0 * static_cast<double>(
                                              m_stats.culledPrimitives) /
                                              static_cast<double>(considered)
                                        : 0.0);
            // Worth showing: a primitive with no authored bounds, or a
            // skinned one, is exempt and always drawn.
            if (m_stats.unculledPrimitives != 0) {
                ImGui::Text("%zu exempt (skinned or no bounds)",
                            m_stats.unculledPrimitives);
            }
        } else {
            ImGui::TextUnformatted("frustum culling off (Render tab)");
        }
        // Summed over the cascades, so this exceeds the primitive count
        // whenever more than one cascade sees the same caster -- which is
        // the cost cascaded shadows actually add to a frame.
        if (m_stats.shadowDraws != 0 || m_stats.shadowCulled != 0) {
            ImGui::Text("%zu shadow draws over all cascades, %zu culled",
                        m_stats.shadowDraws, m_stats.shadowCulled);
        }

        // One draw per primitive, no instancing: this is the number that
        // decides whether a large scene is CPU or GPU bound.
        ImGui::Text("%zu pipeline switches (single/double sided)",
                    m_stats.pipelineSwitches);
        // Consecutive primitives sharing a material rebind nothing, so the
        // gap between this and the draw count is how much the scene's draw
        // order is already doing for us. Sorting draws by material would
        // close it; nothing does that yet.
        ImGui::Text("%zu material table binds of %zu draws",
                    m_stats.materialTableBinds, m_stats.draws);
        // Built once at load, not per frame: no descriptors are copied
        // while rendering. The distinct count is below the material count
        // because a table is deduplicated on its five texture slots, and
        // everything untextured shares one block of five fallbacks.
        ImGui::Text("%zu distinct material tables, %zu descriptors",
                    m_materialTableCount,
                    m_materialTableCount * kMaterialTextureSlotCount);

        ImGui::Separator();
        if (m_lightmapCharts != 0) {
            ImGui::Text("Lightmap: %zu chart(s), %.0f m2 at %.0f texels/m",
                        m_lightmapCharts,
                        static_cast<double>(m_lightmapArea),
                        static_cast<double>(m_lightmapSetup.texelsPerMetre));
            ImGui::Text("   %u x %u atlas, %.1f%% used, %.1f MB",
                        m_lightmap.atlasResolution, m_lightmap.atlasResolution,
                        100.0 * static_cast<double>(m_lightmap.texelsPlaced) /
                            (static_cast<double>(m_lightmap.atlasResolution) *
                             m_lightmap.atlasResolution),
                        static_cast<double>(m_lightmap.atlasResolution) *
                            m_lightmap.atlasResolution / 1048576.0);
            if (!m_lightmap.complete) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                                   "some charts did not fit -- those surfaces "
                                   "fall back to the shadow map");
            }
        } else {
            ImGui::TextUnformatted("Lightmap: nothing baked");
        }
        if (m_lightmapMissingUv != 0) {
            // The number an unwrapper exists to fix. Worth saying plainly:
            // a scene that bakes nothing looks exactly like a scene whose
            // bake is broken.
            ImGui::TextColored(
                ImVec4(1.0f, 0.8f, 0.3f, 1.0f),
                "%zu static instance(s) have no TEXCOORD_1 to bake into",
                m_lightmapMissingUv);
            ImGui::TextUnformatted("   a baked shadow is addressed by the "
                                   "surface, and TEXCOORD_0 tiles");
        }

        ImGui::Separator();
        // Every texture is RGBA8 with a full mip chain and no BC
        // compression, so this is usually the largest single allocation in
        // the process and the first thing a big scene runs out of.
        ImGui::Text("Textures: %.1f MB of video memory",
                    static_cast<double>(m_textureBytes) / (1024.0 * 1024.0));
        if (m_maxTextureDimension != 0) {
            ImGui::Text("capped at %u px; %zu image(s) downscaled",
                        m_maxTextureDimension, m_texturesDownscaled);
            if (m_texturesOverBudget != 0) {
                ImGui::Text("%zu block-compressed image(s) already over the cap",
                            m_texturesOverBudget);
            }
        } else {
            ImGui::TextUnformatted("no texture size cap");
        }
        // A missing image is normal for an MSFT_texture_dds asset, which
        // names a .png source it never shipped for every .dds it did. A
        // FAILED one is not: it means a file was there and could not be
        // read, and it shows as magenta rather than as nothing.
        if (m_imagesMissing != 0) {
            ImGui::Text("%zu image file(s) absent (expected for MSFT_texture_dds)",
                        m_imagesMissing);
        }
        if (m_imagesFailed != 0) {
            ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                               "%zu image(s) failed to decode -- shown magenta",
                               m_imagesFailed);
        }
    }

    void GLTFAdapter::ReleaseStaleDescriptors(uint64_t frameNumber) {
        // Returns descriptors freed on or before frameNumber to their pages.
        // The caller passes a frame number it knows the GPU is done with, so
        // nothing still referenced by an in-flight command list is recycled.
        if (m_srvAllocator) {
            m_srvAllocator->ReleaseStaleDescriptors(frameNumber);
        }
        // The dynamic heap's GPU-visible heaps are recycled by its own Reset()
        // at the top of Render(), which is tied to the command list rather than
        // to this frame-retirement margin.
    }

}