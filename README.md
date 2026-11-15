# hxn-gpu-metal

[![Ash](https://github.com/hx-n/hxn-gpu-metal/actions/workflows/ash.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-metal/actions/workflows/ash.yml)
[![hxcpp](https://github.com/hx-n/hxn-gpu-metal/actions/workflows/hxcpp.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-metal/actions/workflows/hxcpp.yml)
[![CPPIA](https://github.com/hx-n/hxn-gpu-metal/actions/workflows/cppia.yml/badge.svg)](https://github.com/hx-n/hxn-gpu-metal/actions/workflows/cppia.yml)

Metal for Haxe on Ash, HashLink, hxcpp and CPPIA, on macOS and iOS. It wraps
Apple's Metal framework directly, with no wgpu or Naga in between, and keeps
Metal's own conventions: a device makes buffers, textures, libraries and pipelines,
and a command queue's command buffers record compute and render passes.

Shaders come from [hxnsl](https://github.com/hx-n/hxnsl), which compiles HXSL
to MSL during the Haxe build and reports, in `MSL_SLOTS`, the argument index
each resource takes. The driver binds by those indices.

```haxe
import gpu.metal.*;

var device = Device.systemDefault();
var queue = device.newCommandQueue();
var library = device.newLibraryFromSource(Tripled.MSL);    // an hxnsl shader
var pipeline = device.newComputePipeline(library.functionNamed("compute_main"));
var data = device.newBufferWith(input);

var commands = queue.commandBuffer();
var encoder = commands.computeEncoder();
encoder.setPipeline(pipeline);
encoder.setBuffer(data, 0, index);                         // from Tripled.MSL_SLOTS
encoder.dispatchThreads(64, 1, 1, 16, 1, 1);
encoder.end();
commands.completed().then(_ -> trace(data.read(0, 64 * 4)));  // once the GPU is done
commands.commit();
```

[`examples/check/Main.hx`](https://github.com/hx-n/hxn-gpu-metal/blob/main/examples/check/Main.hx)
is the whole program, with a render pass into a texture as well.

## Installing

```
haxelib install hxn-gpu-metal
haxe -lib hxn-gpu-metal -lib hxnsl -main Main -hl main.hl
haxe -lib hxn-gpu-metal -lib hxnsl -main Main -cpp out
```

The haxelib carries the native library prebuilt for macOS on Apple silicon,
for HashLink and hxcpp, and as static libraries for hxcpp on Intel Macs and
for iOS and the iOS simulator. A program needs no Rust. Intel Macs have no
HashLink library yet, because hxn cannot yet build one on an Apple-silicon
machine.

## Classes

All in the package `gpu.metal`:

| Class | |
|---|---|
| `Device` | `systemDefault()`, then buffers, textures, samplers, libraries, pipelines and command queues. |
| `Buffer` | Memory the CPU and GPU share: `write`, `read`. |
| `Texture` | A 2D texture, described by a `TextureDescriptor`: `write`, `read`. |
| `Library`, `Function` | Shader functions, from MSL (`newLibraryFromSource`) or a `.metallib` (`newLibrary`). |
| `ComputePipeline`, `RenderPipeline` | Compiled pipeline states. A render pipeline takes the vertex buffers' layout as `VertexAttribute`s and `VertexLayout`s. |
| `CommandQueue`, `CommandBuffer` | Command buffers, committed and waited on; `present` shows a drawable. |
| `ComputeEncoder`, `RenderEncoder` | Binding by argument index, dispatches and draws. |
| `Layer`, `Drawable` | A `CAMetalLayer` on a window's view, and the drawable to render into. |

Every object holds its Metal object until `close()`. Metal's errors, such as
a shader that does not compile, are thrown as Haxe exceptions.

## Futures

Some calls return an `hxn.Future` (from the `hxn` haxelib, installed with this
one) instead of blocking:

- `CommandBuffer.completed()` settles when the GPU has finished the buffer,
  and fails with Metal's error if it did not. Ask for it before `commit()`.
  `waitUntilCompleted()` is the blocking form.
- `Device.newLibraryFromSourceAsync`, `newComputePipelineAsync` and
  `newRenderPipelineAsync` compile on Metal's threads, so loading shaders does
  not stall a frame.

`await()` waits for the result, and `then` takes a callback. Callbacks run
through `hxn.Loop` on the Haxe event loop; a program with its own frame loop
sets `hxn.Loop.manual` and calls `hxn.Loop.sync()` once a frame.

## Shaders from hxnsl

A shader for this driver implements `gpu.metal.Shader`, which extends
`hxnsl.Shader`. Besides hxnsl's `MSL` and `MSL_SLOTS`, it gets `metallib()`:
the shader compiled to a Metal library during the Haxe build, by
`xcrun metal` and `metallib`, for the SDK the build targets (macOS, or iOS
and the iOS simulator with hxcpp's `-D iphoneos` or `-D iphonesim`).
`Shaders.library` loads it, so the program compiles no shader:

```haxe
class Tinted implements gpu.metal.Shader { static var SRC = { ... }; }

var library = gpu.metal.Shaders.library(device, Tinted.metallib(), Tinted.MSL);
```

Where the build machine has no Metal compiler, the build warns once,
`metallib()` is null, and `Shaders.library` compiles the MSL instead. Xcode
installs the compiler, or `xcodebuild -downloadComponent MetalToolchain`.

- The MSL functions keep hxnsl's entry point names: `vertex_main`,
  `fragment_main` and `compute_main`, the names `MSL_SLOTS` lists them by.
- `MSL_SLOTS` lists, for each entry point, each resource's `kind` (buffer,
  texture or sampler) and `index`. A stage with a storage buffer also has a
  `sizesBuffer`: bind the byte length of each runtime-sized array there, as
  32-bit integers, with `setBytes`.
- Vertex buffers share the buffer argument table with the shader's buffers.
  Bind them at an index no resource uses; the check uses 30.

## Windows

`Layer.attach(device, view, format)` puts a `CAMetalLayer` on a view: the
`NSView` that [hxn-window](https://github.com/hx-n/hxn-window)'s `raw(0)`
returns on macOS, or the `UIView` on iOS. It must run on the main thread.
`nextDrawable()` gives a drawable whose `texture()` a render pass targets, and
`CommandBuffer.present` shows it.

## Building from source

Requires the [hxn](https://github.com/hx-n/hxn) CLI, Rust, Haxe 4.3 and hxnsl.
`hxn check` builds the binding and runs `examples/check` on each runtime
named, and fails unless every run passes and prints the same output:

```
hxn check hl --vm ash
hxn check cpp cppia
```

## Agent skill

[`skills/hxn-gpu-metal/SKILL.md`](https://github.com/hx-n/hxn-gpu-metal/blob/main/skills/hxn-gpu-metal/SKILL.md)
describes the library for coding agents. The haxelib includes it.

## License

Apache-2.0.
