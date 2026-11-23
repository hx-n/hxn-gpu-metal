//! Metal for Haxe: a thin layer over Apple's Metal framework, keeping Metal's conventions.
//! Shaders come from hxnsl as MSL, or as a metallib compiled from it, and are bound by the argument
//! indices hxnsl reports in `MSL_SLOTS`.
//!
//! The `_async` constructors and `CommandBuffer.completed` settle a future from the completion
//! handler Metal calls on its own thread; Haxe receives it through `hxn.Loop`.

use std::ffi::c_void;
use std::ptr::NonNull;

use block2::RcBlock;
use dispatch2::DispatchData;
use hxn_future::Future;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder, MTLBuffer, MTLClearColor,
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLFunction, MTLLibrary, MTLLoadAction, MTLOrigin, MTLPixelFormat as RawPixelFormat,
    MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResourceOptions, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLScissorRect, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
    MTLVertexDescriptor, MTLVertexFormat, MTLVertexStepFunction, MTLViewport,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};

hxn::package!("hxn.gpu.metal");

/// A thread-safe Metal object: a device, queue, resource, pipeline, library or drawable, which
/// Apple documents as usable from any thread.
struct Shared<T: ?Sized>(Retained<T>);
unsafe impl<T: ?Sized> Send for Shared<T> {}
unsafe impl<T: ?Sized> Sync for Shared<T> {}

/// A command buffer or encoder: not thread-safe, so its methods take `&mut self`, and hxn lends
/// it to one call at a time. Releasing it is safe from any thread.
struct Owned<T: ?Sized>(Retained<T>);
unsafe impl<T: ?Sized> Send for Owned<T> {}

fn message(error: Retained<NSError>) -> String {
    error.localizedDescription().to_string()
}

/// The completion handler of an `_async` constructor: what Metal made, or its error.
type Made<P> = RcBlock<dyn Fn(*mut ProtocolObject<P>, *mut NSError)>;

/// A command buffer's completed handler.
type Completed = RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLCommandBuffer>>)>;

/// Metal's error from a completion handler, which passes it unretained.
fn handler_error(error: *mut NSError, otherwise: &str) -> String {
    unsafe { Retained::retain(error) }
        .map(message)
        .unwrap_or_else(|| otherwise.to_string())
}

fn usize_of(value: i64, what: &str) -> Result<usize, String> {
    usize::try_from(value).map_err(|_| format!("{what} cannot be {value}"))
}

/// The pixel formats a texture or render target can have, with Metal's values.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum PixelFormat {
    R8Unorm = 10,
    R32Float = 55,
    Rgba8Unorm = 70,
    Rgba8UnormSrgb = 71,
    Bgra8Unorm = 80,
    Bgra8UnormSrgb = 81,
    Rgba16Float = 115,
    Rgba32Float = 125,
    Depth32Float = 252,
}

impl PixelFormat {
    fn raw(self) -> RawPixelFormat {
        RawPixelFormat(self as usize)
    }

    fn from_raw(raw: RawPixelFormat) -> Option<Self> {
        use PixelFormat::*;
        [
            R8Unorm,
            R32Float,
            Rgba8Unorm,
            Rgba8UnormSrgb,
            Bgra8Unorm,
            Bgra8UnormSrgb,
            Rgba16Float,
            Rgba32Float,
            Depth32Float,
        ]
        .into_iter()
        .find(|f| *f as usize == raw.0)
    }

    fn bytes_per_pixel(self) -> usize {
        use PixelFormat::*;
        match self {
            R8Unorm => 1,
            R32Float | Rgba8Unorm | Rgba8UnormSrgb | Bgra8Unorm | Bgra8UnormSrgb | Depth32Float => {
                4
            }
            Rgba16Float => 8,
            Rgba32Float => 16,
        }
    }
}

/// Where a texture's memory lives. `Shared` is CPU-visible; `Private` is GPU-only.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum StorageMode {
    Shared = 0,
    Managed = 1,
    Private = 2,
}

/// How a texture is used, as `TextureDescriptor.usage` bits.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum TextureUsage {
    ShaderRead = 1,
    ShaderWrite = 2,
    RenderTarget = 4,
}

#[hxn::data]
pub struct TextureDescriptor {
    pub width: i32,
    pub height: i32,
    pub format: PixelFormat,
    /// `TextureUsage` bits, or-ed together.
    pub usage: i32,
    pub storage: StorageMode,
}

#[hxn::data]
#[derive(Clone, Copy)]
pub enum Filter {
    Nearest = 0,
    Linear = 1,
}

#[hxn::data]
#[derive(Clone, Copy)]
pub enum AddressMode {
    ClampToEdge = 0,
    Repeat = 2,
    MirrorRepeat = 3,
}

#[hxn::data]
pub struct SamplerDescriptor {
    pub min_filter: Filter,
    pub mag_filter: Filter,
    pub address: AddressMode,
}

/// A vertex attribute's format, with Metal's values.
#[hxn::data]
#[derive(Clone, Copy)]
pub enum VertexFormat {
    Float = 28,
    Float2 = 29,
    Float3 = 30,
    Float4 = 31,
    Int = 32,
    UInt = 36,
}

/// One vertex attribute: `location` is the shader's (hxnsl's `INPUT_<name>`), `buffer` the
/// argument index its vertex buffer is bound at.
#[hxn::data]
pub struct VertexAttribute {
    pub location: i32,
    pub format: VertexFormat,
    pub offset: i32,
    pub buffer: i32,
}

/// The layout of one vertex buffer: its stride in bytes, and whether it advances per instance.
#[hxn::data]
pub struct VertexLayout {
    pub buffer: i32,
    pub stride: i32,
    pub per_instance: bool,
}

/// How a pipeline blends what it draws into its color target. The factors are Metal's
/// `MTLBlendFactor` values, from 0 `Zero` to 10 `SourceAlphaSaturated`, and the operations its
/// `MTLBlendOperation` values, from 0 `Add` to 4 `Max`, which hxn-gpu's drivers share. With
/// `enabled` false, the pipeline writes its color as it is.
#[hxn::data]
pub struct Blend {
    pub enabled: bool,
    pub color_operation: i32,
    pub color_source: i32,
    pub color_destination: i32,
    pub alpha_operation: i32,
    pub alpha_source: i32,
    pub alpha_destination: i32,
}

#[hxn::data]
#[derive(Clone, Copy)]
pub enum LoadAction {
    DontCare = 0,
    Load = 1,
    Clear = 2,
}

#[hxn::data]
#[derive(Clone, Copy)]
pub enum Primitive {
    Point = 0,
    Line = 1,
    LineStrip = 2,
    Triangle = 3,
    TriangleStrip = 4,
}

/// A GPU, and the maker of every other Metal object.
#[hxn::interface]
pub struct Device {
    device: Shared<ProtocolObject<dyn MTLDevice>>,
}

#[hxn::methods]
impl Device {
    /// The system's default GPU. Fails where there is none, such as a virtual machine without one.
    pub fn system_default() -> Result<Device, String> {
        hxn::log::debug!(target: "hxn.gpu.metal", "opening the system default Metal device");
        let device = MTLCreateSystemDefaultDevice().ok_or("this machine has no Metal device")?;
        Ok(Device {
            device: Shared(device),
        })
    }

    pub fn name(&self) -> String {
        self.device.0.name().to_string()
    }

    /// A buffer of `length` bytes in memory the CPU and GPU share, zeroed.
    pub fn new_buffer(&self, length: i64) -> Result<Buffer, String> {
        let length = usize_of(length, "a buffer's length")?.max(1);
        let buffer = self
            .device
            .0
            .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
            .ok_or("Metal could not allocate the buffer")?;
        Ok(Buffer {
            buffer: Shared(buffer),
        })
    }

    /// A shared buffer holding a copy of `bytes`.
    pub fn new_buffer_with(&self, bytes: &[u8]) -> Result<Buffer, String> {
        let buffer = self.new_buffer(bytes.len() as i64)?;
        buffer.write(0, bytes)?;
        Ok(buffer)
    }

    /// A library from MSL source, compiled now: hxnsl's `MSL`.
    pub fn new_library_from_source(&self, msl: &str) -> Result<Library, String> {
        let library = self
            .device
            .0
            .newLibraryWithSource_options_error(&NSString::from_str(msl), None)
            .map_err(message)?;
        Ok(Library {
            library: Shared(library),
        })
    }

    /// A library from MSL source, compiled on Metal's threads; the future settles when it is.
    pub fn new_library_from_source_async(&self, msl: &str) -> Future<Library> {
        let (completer, future) = hxn_future::pending::<Library>();
        let handler: Made<dyn MTLLibrary> =
            RcBlock::new(
                move |library, error| match unsafe { Retained::retain(library) } {
                    Some(library) => {
                        completer.resolve(Library {
                            library: Shared(library),
                        });
                    }
                    None => {
                        completer.reject(handler_error(error, "Metal compiled no library"));
                    }
                },
            );
        unsafe {
            self.device
                .0
                .newLibraryWithSource_options_completionHandler(
                    &NSString::from_str(msl),
                    None,
                    RcBlock::as_ptr(&handler),
                )
        };
        future
    }

    /// A library from a compiled `.metallib`.
    pub fn new_library(&self, metallib: &[u8]) -> Result<Library, String> {
        let data = DispatchData::from_bytes(metallib);
        let library = self
            .device
            .0
            .newLibraryWithData_error(&data)
            .map_err(message)?;
        Ok(Library {
            library: Shared(library),
        })
    }

    pub fn new_command_queue(&self) -> Result<CommandQueue, String> {
        let queue = self
            .device
            .0
            .newCommandQueue()
            .ok_or("Metal could not make a command queue")?;
        Ok(CommandQueue {
            queue: Shared(queue),
        })
    }

    pub fn new_compute_pipeline(&self, function: &Function) -> Result<ComputePipeline, String> {
        let pipeline = self
            .device
            .0
            .newComputePipelineStateWithFunction_error(&function.function.0)
            .map_err(message)?;
        Ok(ComputePipeline {
            pipeline: Shared(pipeline),
        })
    }

    /// A compute pipeline, compiled on Metal's threads; the future settles when it is.
    pub fn new_compute_pipeline_async(&self, function: &Function) -> Future<ComputePipeline> {
        let (completer, future) = hxn_future::pending::<ComputePipeline>();
        let handler: Made<dyn MTLComputePipelineState> = RcBlock::new(
            move |pipeline, error| match unsafe { Retained::retain(pipeline) } {
                Some(pipeline) => {
                    completer.resolve(ComputePipeline {
                        pipeline: Shared(pipeline),
                    });
                }
                None => {
                    completer.reject(handler_error(error, "Metal made no compute pipeline"));
                }
            },
        );
        unsafe {
            self.device
                .0
                .newComputePipelineStateWithFunction_completionHandler(
                    &function.function.0,
                    RcBlock::as_ptr(&handler),
                )
        };
        future
    }

    /// A render pipeline drawing to one color target of `format`, with vertex buffers laid out
    /// as `attributes` and `layouts` describe (both empty when the vertex stage reads none),
    /// blending as `blend` describes.
    pub fn new_render_pipeline(
        &self,
        vertex: &Function,
        fragment: &Function,
        format: PixelFormat,
        attributes: Vec<VertexAttribute>,
        layouts: Vec<VertexLayout>,
        blend: Blend,
    ) -> Result<RenderPipeline, String> {
        let descriptor =
            render_descriptor(vertex, fragment, format, &attributes, &layouts, &blend)?;
        let pipeline = self
            .device
            .0
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(message)?;
        Ok(RenderPipeline {
            pipeline: Shared(pipeline),
        })
    }

    /// `new_render_pipeline`, compiled on Metal's threads; the future settles when it is. Fails
    /// at once for a vertex layout Metal cannot take.
    pub fn new_render_pipeline_async(
        &self,
        vertex: &Function,
        fragment: &Function,
        format: PixelFormat,
        attributes: Vec<VertexAttribute>,
        layouts: Vec<VertexLayout>,
        blend: Blend,
    ) -> Result<Future<RenderPipeline>, String> {
        let descriptor =
            render_descriptor(vertex, fragment, format, &attributes, &layouts, &blend)?;
        let (completer, future) = hxn_future::pending::<RenderPipeline>();
        let handler: Made<dyn MTLRenderPipelineState> = RcBlock::new(
            move |pipeline, error| match unsafe { Retained::retain(pipeline) } {
                Some(pipeline) => {
                    completer.resolve(RenderPipeline {
                        pipeline: Shared(pipeline),
                    });
                }
                None => {
                    completer.reject(handler_error(error, "Metal made no render pipeline"));
                }
            },
        );
        unsafe {
            self.device
                .0
                .newRenderPipelineStateWithDescriptor_completionHandler(
                    &descriptor,
                    RcBlock::as_ptr(&handler),
                )
        };
        Ok(future)
    }

    pub fn new_texture(&self, descriptor: TextureDescriptor) -> Result<Texture, String> {
        let width = usize_of(descriptor.width as i64, "a texture's width")?;
        let height = usize_of(descriptor.height as i64, "a texture's height")?;
        let d = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                descriptor.format.raw(),
                width,
                height,
                false,
            )
        };
        d.setUsage(MTLTextureUsage(descriptor.usage as usize));
        d.setStorageMode(MTLStorageMode(descriptor.storage as usize));
        let texture = self
            .device
            .0
            .newTextureWithDescriptor(&d)
            .ok_or("Metal could not make the texture")?;
        Ok(Texture {
            texture: Shared(texture),
        })
    }

    pub fn new_sampler(&self, descriptor: SamplerDescriptor) -> Result<Sampler, String> {
        let filter = |f: Filter| match f {
            Filter::Nearest => MTLSamplerMinMagFilter::Nearest,
            Filter::Linear => MTLSamplerMinMagFilter::Linear,
        };
        let d = MTLSamplerDescriptor::new();
        d.setMinFilter(filter(descriptor.min_filter));
        d.setMagFilter(filter(descriptor.mag_filter));
        let address = MTLSamplerAddressMode(descriptor.address as usize);
        d.setSAddressMode(address);
        d.setTAddressMode(address);
        d.setRAddressMode(address);
        let sampler = self
            .device
            .0
            .newSamplerStateWithDescriptor(&d)
            .ok_or("Metal could not make the sampler")?;
        Ok(Sampler {
            sampler: Shared(sampler),
        })
    }
}

/// The descriptor `new_render_pipeline` and its async form compile.
fn render_descriptor(
    vertex: &Function,
    fragment: &Function,
    format: PixelFormat,
    attributes: &[VertexAttribute],
    layouts: &[VertexLayout],
    blend: &Blend,
) -> Result<Retained<MTLRenderPipelineDescriptor>, String> {
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setVertexFunction(Some(&vertex.function.0));
    descriptor.setFragmentFunction(Some(&fragment.function.0));
    let color = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color.setPixelFormat(format.raw());
    if blend.enabled {
        let factor = |code: i32, what: &str| -> Result<MTLBlendFactor, String> {
            if (0..=10).contains(&code) {
                Ok(MTLBlendFactor(code as usize))
            } else {
                Err(format!("{what} is {code}, and a blend factor is 0 to 10"))
            }
        };
        let operation = |code: i32, what: &str| -> Result<MTLBlendOperation, String> {
            if (0..=4).contains(&code) {
                Ok(MTLBlendOperation(code as usize))
            } else {
                Err(format!("{what} is {code}, and a blend operation is 0 to 4"))
            }
        };
        color.setBlendingEnabled(true);
        color.setRgbBlendOperation(operation(blend.color_operation, "the color operation")?);
        color.setSourceRGBBlendFactor(factor(blend.color_source, "the color source")?);
        color.setDestinationRGBBlendFactor(factor(
            blend.color_destination,
            "the color destination",
        )?);
        color.setAlphaBlendOperation(operation(blend.alpha_operation, "the alpha operation")?);
        color.setSourceAlphaBlendFactor(factor(blend.alpha_source, "the alpha source")?);
        color.setDestinationAlphaBlendFactor(factor(
            blend.alpha_destination,
            "the alpha destination",
        )?);
    }
    if !attributes.is_empty() {
        let vertices = MTLVertexDescriptor::new();
        for a in attributes {
            let attribute = unsafe {
                vertices
                    .attributes()
                    .objectAtIndexedSubscript(usize_of(a.location as i64, "a location")?)
            };
            attribute.setFormat(MTLVertexFormat(a.format as isize as usize));
            unsafe { attribute.setOffset(usize_of(a.offset as i64, "an offset")?) };
            unsafe { attribute.setBufferIndex(usize_of(a.buffer as i64, "a buffer index")?) };
        }
        for l in layouts {
            let layout = unsafe {
                vertices
                    .layouts()
                    .objectAtIndexedSubscript(usize_of(l.buffer as i64, "a buffer index")?)
            };
            unsafe { layout.setStride(usize_of(l.stride as i64, "a stride")?) };
            layout.setStepFunction(if l.per_instance {
                MTLVertexStepFunction::PerInstance
            } else {
                MTLVertexStepFunction::PerVertex
            });
        }
        descriptor.setVertexDescriptor(Some(&vertices));
    }
    Ok(descriptor)
}

/// GPU memory the CPU can read and write.
#[hxn::interface]
pub struct Buffer {
    buffer: Shared<ProtocolObject<dyn MTLBuffer>>,
}

#[hxn::methods]
impl Buffer {
    pub fn length(&self) -> i64 {
        self.buffer.0.length() as i64
    }

    /// Copies `bytes` into the buffer at `offset`.
    pub fn write(&self, offset: i64, bytes: &[u8]) -> Result<(), String> {
        let start = self.range(offset, bytes.len())?;
        let base = self.buffer.0.contents().as_ptr().cast::<u8>();
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), base.add(start), bytes.len()) };
        Ok(())
    }

    /// `length` bytes from `offset`.
    pub fn read(&self, offset: i64, length: i64) -> Result<Vec<u8>, String> {
        let length = usize_of(length, "a length")?;
        let start = self.range(offset, length)?;
        let base = self.buffer.0.contents().as_ptr().cast::<u8>();
        Ok(unsafe { std::slice::from_raw_parts(base.add(start), length) }.to_vec())
    }
}

impl Buffer {
    fn range(&self, offset: i64, length: usize) -> Result<usize, String> {
        let start = usize_of(offset, "an offset")?;
        let size = self.buffer.0.length();
        if start.checked_add(length).is_none_or(|end| end > size) {
            return Err(format!(
                "{length} bytes at {start} run past the buffer's {size}"
            ));
        }
        Ok(start)
    }
}

/// A 2D texture.
#[hxn::interface]
pub struct Texture {
    texture: Shared<ProtocolObject<dyn MTLTexture>>,
}

#[hxn::methods]
impl Texture {
    pub fn width(&self) -> i32 {
        self.texture.0.width() as i32
    }

    pub fn height(&self) -> i32 {
        self.texture.0.height() as i32
    }

    /// Its format, or null for one this API does not name.
    pub fn format(&self) -> Option<PixelFormat> {
        PixelFormat::from_raw(self.texture.0.pixelFormat())
    }

    /// Replaces the whole texture with `bytes`, rows `bytes_per_row` apart. The texture's storage
    /// must be shared or managed.
    pub fn write(&self, bytes: &[u8], bytes_per_row: i32) -> Result<(), String> {
        let row = usize_of(bytes_per_row as i64, "bytes per row")?;
        let (region, needed) = self.whole(row)?;
        if bytes.len() < needed {
            return Err(format!(
                "{} bytes cannot fill a texture that needs {needed}",
                bytes.len()
            ));
        }
        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void).ok_or("no bytes")?;
        unsafe {
            self.texture
                .0
                .replaceRegion_mipmapLevel_withBytes_bytesPerRow(region, 0, pointer, row)
        };
        Ok(())
    }

    /// Replaces the `width` by `height` pixels at `x`, `y` with `bytes`, rows `bytes_per_row`
    /// apart. The texture's storage must be shared or managed.
    pub fn write_region(
        &self,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        bytes: &[u8],
        bytes_per_row: i32,
    ) -> Result<(), String> {
        let row = usize_of(bytes_per_row as i64, "bytes per row")?;
        let (x, y) = (usize_of(x as i64, "x")?, usize_of(y as i64, "y")?);
        let (width, height) = (
            usize_of(width as i64, "a width")?,
            usize_of(height as i64, "a height")?,
        );
        if width == 0 || height == 0 {
            return Err("a region is at least one pixel".to_string());
        }
        if x + width > self.texture.0.width() || y + height > self.texture.0.height() {
            return Err(format!(
                "a {width}x{height} region at {x}, {y} goes past the edge of the texture"
            ));
        }
        let pixel = self.format().map_or(4, PixelFormat::bytes_per_pixel);
        let packed = width * pixel;
        if row < packed {
            return Err(format!(
                "rows of {width} pixels take {packed} bytes, more than bytes per row {row}"
            ));
        }
        let needed = row
            .checked_mul(height - 1)
            .and_then(|n| n.checked_add(packed))
            .ok_or("the region is too large")?;
        if bytes.len() < needed {
            return Err(format!(
                "{} bytes cannot fill a region that needs {needed}",
                bytes.len()
            ));
        }
        let region = MTLRegion {
            origin: MTLOrigin { x, y, z: 0 },
            size: MTLSize {
                width,
                height,
                depth: 1,
            },
        };
        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void).ok_or("no bytes")?;
        unsafe {
            self.texture
                .0
                .replaceRegion_mipmapLevel_withBytes_bytesPerRow(region, 0, pointer, row)
        };
        Ok(())
    }

    /// The whole texture's bytes, rows `bytes_per_row` apart. A managed texture must be
    /// synchronized first (`CommandBuffer.synchronize`) after the GPU wrote it.
    pub fn read(&self, bytes_per_row: i32) -> Result<Vec<u8>, String> {
        let row = usize_of(bytes_per_row as i64, "bytes per row")?;
        let (region, needed) = self.whole(row)?;
        let mut out = vec![0u8; needed];
        let pointer = NonNull::new(out.as_mut_ptr().cast::<c_void>()).ok_or("no bytes")?;
        unsafe {
            self.texture
                .0
                .getBytes_bytesPerRow_fromRegion_mipmapLevel(pointer, row, region, 0)
        };
        Ok(out)
    }
}

impl Texture {
    fn whole(&self, row: usize) -> Result<(MTLRegion, usize), String> {
        let (width, height) = (self.texture.0.width(), self.texture.0.height());
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize {
                width,
                height,
                depth: 1,
            },
        };
        let needed = row.checked_mul(height).ok_or("the texture is too large")?;
        Ok((region, needed))
    }
}

#[hxn::interface]
pub struct Sampler {
    sampler: Shared<ProtocolObject<dyn MTLSamplerState>>,
}

#[hxn::methods]
impl Sampler {}

/// Compiled shader functions.
#[hxn::interface]
pub struct Library {
    library: Shared<ProtocolObject<dyn MTLLibrary>>,
}

#[hxn::methods]
impl Library {
    /// The function `name`, as the MSL names it.
    pub fn function_named(&self, name: &str) -> Result<Function, String> {
        let function = self
            .library
            .0
            .newFunctionWithName(&NSString::from_str(name))
            .ok_or_else(|| format!("the library has no function {name}"))?;
        Ok(Function {
            function: Shared(function),
        })
    }

    /// The names of the library's functions.
    pub fn function_names(&self) -> Vec<String> {
        self.library
            .0
            .functionNames()
            .iter()
            .map(|n| n.to_string())
            .collect()
    }
}

#[hxn::interface]
pub struct Function {
    function: Shared<ProtocolObject<dyn MTLFunction>>,
}

#[hxn::methods]
impl Function {
    pub fn name(&self) -> String {
        self.function.0.name().to_string()
    }
}

#[hxn::interface]
pub struct ComputePipeline {
    pipeline: Shared<ProtocolObject<dyn MTLComputePipelineState>>,
}

#[hxn::methods]
impl ComputePipeline {
    /// The SIMD width: a threadgroup size that is a multiple of it fills the GPU's lanes.
    pub fn thread_execution_width(&self) -> i32 {
        self.pipeline.0.threadExecutionWidth() as i32
    }

    pub fn max_total_threads_per_threadgroup(&self) -> i32 {
        self.pipeline.0.maxTotalThreadsPerThreadgroup() as i32
    }
}

#[hxn::interface]
pub struct RenderPipeline {
    pipeline: Shared<ProtocolObject<dyn MTLRenderPipelineState>>,
}

#[hxn::methods]
impl RenderPipeline {}

#[hxn::interface]
pub struct CommandQueue {
    queue: Shared<ProtocolObject<dyn MTLCommandQueue>>,
}

#[hxn::methods]
impl CommandQueue {
    pub fn command_buffer(&self) -> Result<CommandBuffer, String> {
        let buffer = self
            .queue
            .0
            .commandBuffer()
            .ok_or("Metal could not make a command buffer")?;
        Ok(CommandBuffer {
            buffer: Owned(buffer),
        })
    }
}

/// Commands recorded through its encoders, then committed to the GPU.
#[hxn::interface]
pub struct CommandBuffer {
    buffer: Owned<ProtocolObject<dyn MTLCommandBuffer>>,
}

#[hxn::methods]
impl CommandBuffer {
    pub fn compute_encoder(&mut self) -> Result<ComputeEncoder, String> {
        let encoder = self
            .buffer
            .0
            .computeCommandEncoder()
            .ok_or("Metal could not make a compute encoder")?;
        Ok(ComputeEncoder {
            encoder: Owned(encoder),
        })
    }

    /// A render pass to `target`, its one color attachment, which `load` starts from: the
    /// target as it is, cleared to `red`, `green`, `blue`, `alpha`, or undefined.
    pub fn render_encoder(
        &mut self,
        target: &Texture,
        load: LoadAction,
        red: f64,
        green: f64,
        blue: f64,
        alpha: f64,
    ) -> Result<RenderEncoder, String> {
        let pass = MTLRenderPassDescriptor::new();
        let color = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        color.setTexture(Some(&target.texture.0));
        color.setLoadAction(MTLLoadAction(load as usize));
        color.setStoreAction(MTLStoreAction::Store);
        color.setClearColor(MTLClearColor {
            red,
            green,
            blue,
            alpha,
        });
        let encoder = self
            .buffer
            .0
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or("Metal could not make a render encoder")?;
        Ok(RenderEncoder {
            encoder: Owned(encoder),
        })
    }

    /// Makes a managed texture's GPU writes visible to `Texture.read` once this buffer completes.
    pub fn synchronize(&mut self, texture: &Texture) -> Result<(), String> {
        let blit = self
            .buffer
            .0
            .blitCommandEncoder()
            .ok_or("Metal could not make a blit encoder")?;
        let resource = ProtocolObject::from_ref(&*texture.texture.0);
        blit.synchronizeResource(resource);
        blit.endEncoding();
        Ok(())
    }

    /// Settles when the GPU has finished the buffer, or fails with Metal's error. Ask for it
    /// before `commit`; Metal takes no completion handler after.
    pub fn completed(&mut self) -> Result<Future<()>, String> {
        if self.buffer.0.status() != MTLCommandBufferStatus::NotEnqueued
            && self.buffer.0.status() != MTLCommandBufferStatus::Enqueued
        {
            return Err("ask for completed() before commit()".into());
        }
        let (completer, future) = hxn_future::pending::<()>();
        let handler: Completed = RcBlock::new(
            move |buffer: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
                let buffer = unsafe { buffer.as_ref() };
                if buffer.status() == MTLCommandBufferStatus::Error {
                    completer.reject(
                        buffer
                            .error()
                            .map(message)
                            .unwrap_or_else(|| "the command buffer failed".into()),
                    );
                } else {
                    completer.resolve(());
                }
            },
        );
        unsafe { self.buffer.0.addCompletedHandler(RcBlock::as_ptr(&handler)) };
        Ok(future)
    }

    /// Presents `drawable` when the buffer's commands complete.
    pub fn present(&mut self, drawable: &Drawable) {
        self.buffer
            .0
            .presentDrawable(ProtocolObject::from_ref(&*drawable.drawable.0));
    }

    pub fn commit(&mut self) {
        self.buffer.0.commit();
    }

    /// Waits for the GPU to finish the buffer, and fails with Metal's error if it did not.
    #[hxn(blocking)]
    pub fn wait_until_completed(&mut self) -> Result<(), String> {
        self.buffer.0.waitUntilCompleted();
        if self.buffer.0.status() == MTLCommandBufferStatus::Error {
            return Err(self
                .buffer
                .0
                .error()
                .map(message)
                .unwrap_or_else(|| "the command buffer failed".into()));
        }
        Ok(())
    }
}

/// Records compute commands. `end` closes it before the next encoder or `commit`.
#[hxn::interface]
pub struct ComputeEncoder {
    encoder: Owned<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

#[hxn::methods]
impl ComputeEncoder {
    pub fn set_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.encoder.0.setComputePipelineState(&pipeline.pipeline.0);
    }

    /// Binds `buffer` at argument `index`: the kernel's `MSL_SLOTS` buffer index.
    pub fn set_buffer(&mut self, buffer: &Buffer, offset: i64, index: i32) -> Result<(), String> {
        let (offset, index) = (
            usize_of(offset, "an offset")?,
            usize_of(index as i64, "an index")?,
        );
        unsafe {
            self.encoder
                .0
                .setBuffer_offset_atIndex(Some(&buffer.buffer.0), offset, index)
        };
        Ok(())
    }

    /// Binds a copy of `bytes`, up to 4 KiB, at argument `index`.
    pub fn set_bytes(&mut self, bytes: &[u8], index: i32) -> Result<(), String> {
        let index = usize_of(index as i64, "an index")?;
        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void).ok_or("no bytes")?;
        unsafe {
            self.encoder
                .0
                .setBytes_length_atIndex(pointer, bytes.len(), index)
        };
        Ok(())
    }

    pub fn set_texture(&mut self, texture: &Texture, index: i32) -> Result<(), String> {
        let index = usize_of(index as i64, "an index")?;
        unsafe {
            self.encoder
                .0
                .setTexture_atIndex(Some(&texture.texture.0), index)
        };
        Ok(())
    }

    pub fn set_sampler(&mut self, sampler: &Sampler, index: i32) -> Result<(), String> {
        let index = usize_of(index as i64, "an index")?;
        unsafe {
            self.encoder
                .0
                .setSamplerState_atIndex(Some(&sampler.sampler.0), index)
        };
        Ok(())
    }

    /// Runs `x` by `y` by `z` threads in all, in threadgroups of `group_x` by `group_y` by
    /// `group_z`; the last threadgroup in each dimension may be partial.
    pub fn dispatch_threads(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        group_x: i32,
        group_y: i32,
        group_z: i32,
    ) -> Result<(), String> {
        let size = |w: i32, h: i32, d: i32| -> Result<MTLSize, String> {
            Ok(MTLSize {
                width: usize_of(w as i64, "a size")?,
                height: usize_of(h as i64, "a size")?,
                depth: usize_of(d as i64, "a size")?,
            })
        };
        self.encoder.0.dispatchThreads_threadsPerThreadgroup(
            size(x, y, z)?,
            size(group_x, group_y, group_z)?,
        );
        Ok(())
    }

    /// Runs `x` by `y` by `z` threadgroups of `group_x` by `group_y` by `group_z` threads.
    pub fn dispatch_threadgroups(
        &mut self,
        x: i32,
        y: i32,
        z: i32,
        group_x: i32,
        group_y: i32,
        group_z: i32,
    ) -> Result<(), String> {
        let size = |w: i32, h: i32, d: i32| -> Result<MTLSize, String> {
            Ok(MTLSize {
                width: usize_of(w as i64, "a size")?,
                height: usize_of(h as i64, "a size")?,
                depth: usize_of(d as i64, "a size")?,
            })
        };
        self.encoder.0.dispatchThreadgroups_threadsPerThreadgroup(
            size(x, y, z)?,
            size(group_x, group_y, group_z)?,
        );
        Ok(())
    }

    pub fn end(&mut self) {
        self.encoder.0.endEncoding();
    }
}

/// Records draw commands for one render pass. `end` closes it.
#[hxn::interface]
pub struct RenderEncoder {
    encoder: Owned<ProtocolObject<dyn MTLRenderCommandEncoder>>,
}

#[hxn::methods]
impl RenderEncoder {
    pub fn set_pipeline(&mut self, pipeline: &RenderPipeline) {
        self.encoder.0.setRenderPipelineState(&pipeline.pipeline.0);
    }

    pub fn set_vertex_buffer(
        &mut self,
        buffer: &Buffer,
        offset: i64,
        index: i32,
    ) -> Result<(), String> {
        let (offset, index) = (
            usize_of(offset, "an offset")?,
            usize_of(index as i64, "an index")?,
        );
        unsafe {
            self.encoder
                .0
                .setVertexBuffer_offset_atIndex(Some(&buffer.buffer.0), offset, index)
        };
        Ok(())
    }

    pub fn set_fragment_buffer(
        &mut self,
        buffer: &Buffer,
        offset: i64,
        index: i32,
    ) -> Result<(), String> {
        let (offset, index) = (
            usize_of(offset, "an offset")?,
            usize_of(index as i64, "an index")?,
        );
        unsafe {
            self.encoder
                .0
                .setFragmentBuffer_offset_atIndex(Some(&buffer.buffer.0), offset, index)
        };
        Ok(())
    }

    /// Binds a copy of `bytes`, up to 4 KiB, at the vertex stage's argument `index`.
    pub fn set_vertex_bytes(&mut self, bytes: &[u8], index: i32) -> Result<(), String> {
        let index = usize_of(index as i64, "an index")?;
        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void).ok_or("no bytes")?;
        unsafe {
            self.encoder
                .0
                .setVertexBytes_length_atIndex(pointer, bytes.len(), index)
        };
        Ok(())
    }

    /// Binds a copy of `bytes`, up to 4 KiB, at the fragment stage's argument `index`.
    pub fn set_fragment_bytes(&mut self, bytes: &[u8], index: i32) -> Result<(), String> {
        let index = usize_of(index as i64, "an index")?;
        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void).ok_or("no bytes")?;
        unsafe {
            self.encoder
                .0
                .setFragmentBytes_length_atIndex(pointer, bytes.len(), index)
        };
        Ok(())
    }

    pub fn set_fragment_texture(&mut self, texture: &Texture, index: i32) -> Result<(), String> {
        let index = usize_of(index as i64, "an index")?;
        unsafe {
            self.encoder
                .0
                .setFragmentTexture_atIndex(Some(&texture.texture.0), index)
        };
        Ok(())
    }

    pub fn set_fragment_sampler(&mut self, sampler: &Sampler, index: i32) -> Result<(), String> {
        let index = usize_of(index as i64, "an index")?;
        unsafe {
            self.encoder
                .0
                .setFragmentSamplerState_atIndex(Some(&sampler.sampler.0), index)
        };
        Ok(())
    }

    pub fn set_viewport(&mut self, x: f64, y: f64, width: f64, height: f64) {
        self.encoder.0.setViewport(MTLViewport {
            originX: x,
            originY: y,
            width,
            height,
            znear: 0.0,
            zfar: 1.0,
        });
    }

    /// Draws only inside the `width` by `height` pixels at `x`, `y` of the target.
    pub fn set_scissor(&mut self, x: i32, y: i32, width: i32, height: i32) -> Result<(), String> {
        self.encoder.0.setScissorRect(MTLScissorRect {
            x: usize_of(x as i64, "x")?,
            y: usize_of(y as i64, "y")?,
            width: usize_of(width as i64, "a width")?,
            height: usize_of(height as i64, "a height")?,
        });
        Ok(())
    }

    /// Draws `count` vertices from `start`, `instances` times.
    pub fn draw(
        &mut self,
        primitive: Primitive,
        start: i32,
        count: i32,
        instances: i32,
    ) -> Result<(), String> {
        unsafe {
            self.encoder
                .0
                .drawPrimitives_vertexStart_vertexCount_instanceCount(
                    MTLPrimitiveType(primitive as usize),
                    usize_of(start as i64, "a start")?,
                    usize_of(count as i64, "a count")?,
                    usize_of(instances as i64, "an instance count")?,
                )
        };
        Ok(())
    }

    pub fn end(&mut self) {
        self.encoder.0.endEncoding();
    }
}

/// A CAMetalLayer on a window's view, whose drawables a render pass targets.
#[hxn::interface]
pub struct Layer {
    layer: Shared<CAMetalLayer>,
}

#[hxn::methods]
impl Layer {
    /// A layer drawing with `device` into the view `view`: an `NSView` on macOS (hxn-window's
    /// `raw(0)` on AppKit) or a `UIView` on iOS (`raw(0)` on UIKit). Only on the main thread.
    pub fn attach(device: &Device, view: i64, format: PixelFormat) -> Result<Layer, String> {
        if objc2::MainThreadMarker::new().is_none() {
            return Err("a layer can only be attached on the main thread".into());
        }
        let pointer = NonNull::new(view as usize as *mut c_void).ok_or("the view is null")?;
        // The view holds the layer from here on; `into_raw` hands over this reference to it.
        let attached = unsafe {
            #[cfg(target_os = "macos")]
            {
                raw_window_metal::Layer::from_ns_view(pointer)
            }
            #[cfg(not(target_os = "macos"))]
            {
                raw_window_metal::Layer::from_ui_view(pointer)
            }
        };
        let layer: Retained<CAMetalLayer> =
            unsafe { Retained::from_raw(attached.into_raw().as_ptr().cast()) }
                .ok_or("the view gave no layer")?;
        layer.setDevice(Some(&device.device.0));
        layer.setPixelFormat(format.raw());
        Ok(Layer {
            layer: Shared(layer),
        })
    }

    /// Sets the size, in pixels, of the drawables the layer gives.
    pub fn set_drawable_size(&self, width: f64, height: f64) {
        self.layer
            .0
            .setDrawableSize(objc2_core_foundation::CGSize { width, height });
    }

    /// The next drawable to render into, or null when none is free in time.
    pub fn next_drawable(&self) -> Option<Drawable> {
        let drawable = self.layer.0.nextDrawable()?;
        Some(Drawable {
            drawable: Shared(drawable),
        })
    }
}

#[hxn::interface]
pub struct Drawable {
    drawable: Shared<ProtocolObject<dyn CAMetalDrawable>>,
}

#[hxn::methods]
impl Drawable {
    /// The texture to render into.
    pub fn texture(&self) -> Texture {
        Texture {
            texture: Shared(self.drawable.0.texture()),
        }
    }
}
