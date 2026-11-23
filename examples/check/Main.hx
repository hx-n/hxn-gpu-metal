import hxn.gpu.metal.*;
import haxe.io.Bytes;
import hxnsl.Layout.MslEntry;
import Shaders;

/**
	What CI runs: a compute shader over a buffer, and render passes into a
	texture, without blending and with alpha blending, each read back and
	checked. It needs a Metal device, and prints
	the same lines on every runtime.
**/
class Main {
	/** Where `entryPoint` reads the resource at `group` and `binding`, from hxnsl's `MSL_SLOTS`. **/
	static function slot(entries:Array<MslEntry>, entryPoint:String, group:Int, binding:Int):Int {
		var entry = Lambda.find(entries, e -> e.entryPoint == entryPoint);
		return Lambda.find(entry.resources, r -> r.group == group && r.binding == binding).index;
	}

	static function closeAll(objects:Array<{function close():Void;}>) {
		for (o in objects)
			o.close();
	}

	static function check(ok:Bool, message:String) {
		if (!ok)
			throw message;
	}

	static function main() {
		var device = try Device.systemDefault() catch (e:haxe.Exception) {
			Sys.println('no Metal device: ${e.message}');
			return;
		}
		var queue = device.newCommandQueue();
		// The compute pass completes in a callback from the Haxe event loop, after main
		// returns, and the render pass runs from there.
		compute(device, queue, () -> {
			render(device, queue);
			queue.close();
			device.close();
		});
		Sys.println("compute: committed");
	}

	static function compute(device:Device, queue:CommandQueue, next:() -> Void) {
		// Compiled on Metal's threads; await() takes each result when it is ready.
		var library = device.newLibraryFromSourceAsync(Tripled.MSL).await();
		var kernel = library.functionNamed("compute_main");
		var pipeline = device.newComputePipelineAsync(kernel).await();
		var count = 64;
		var input = Bytes.alloc(count * 4);
		for (i in 0...count)
			input.setInt32(i * 4, i);
		var data = device.newBufferWith(input);

		var commands = queue.commandBuffer();
		var encoder = commands.computeEncoder();
		encoder.setPipeline(pipeline);
		encoder.setBuffer(data, 0, slot(Tripled.MSL_SLOTS, "compute_main", Tripled.BUFFER_data_GROUP, Tripled.BUFFER_data));
		// The lengths of runtime-sized arrays, which the kernel reads for bounds.
		var entry = Lambda.find(Tripled.MSL_SLOTS, e -> e.entryPoint == "compute_main");
		if (entry.sizesBuffer != null) {
			var sizes = Bytes.alloc(4);
			sizes.setInt32(0, count * 4);
			encoder.setBytes(sizes, entry.sizesBuffer);
		}
		var group = Tripled.ENTRY_POINTS[0].workgroup[0];
		encoder.dispatchThreads(count, 1, 1, group, 1, 1);
		encoder.end();
		var done = commands.completed();
		commands.commit();
		// Metal takes no completion handler once the buffer is committed.
		var refused = try {
			commands.completed();
			false;
		} catch (e:haxe.Exception) true;
		check(refused, "completed() was accepted after commit()");

		done.then(_ -> {
			var output = data.read(0, count * 4);
			for (i in 0...count)
				check(output.getInt32(i * 4) == i * 3 + 1, 'value $i is ${output.getInt32(i * 4)}, not ${i * 3 + 1}');
			Sys.println('compute: $count values tripled plus one, the last ${output.getInt32((count - 1) * 4)}');
			closeAll([encoder, commands, data, pipeline, kernel, library]);
			next();
		}, e -> {
			Sys.println('compute failed: ${e.message}');
			hxn.Loop.exit(1);
		});
	}

	// Blending is off, so the quad's color replaces the target's.
	static final REPLACE:Blend = {
		enabled: false,
		colorOperation: 0,
		colorSource: 1,
		colorDestination: 0,
		alphaOperation: 0,
		alphaSource: 1,
		alphaDestination: 0
	};

	// Straight alpha blends with color = source * source alpha + destination * (1 - source
	// alpha) and alpha = source + destination * (1 - source alpha). It uses Metal's factors 4
	// (SourceAlpha), 5 (OneMinusSourceAlpha) and 1 (One), and its operation 0 (Add).
	static final ALPHA:Blend = {
		enabled: true,
		colorOperation: 0,
		colorSource: 4,
		colorDestination: 5,
		alphaOperation: 0,
		alphaSource: 1,
		alphaDestination: 5
	};

	static function render(device:Device, queue:CommandQueue) {
		var tint = [0.25, 0.5, 0.75, 1.0];
		var pixels = quad(device, queue, tint, [0.0, 0.0, 0.0, 0.0], REPLACE);
		var expected = [for (c in tint) Math.round(c * 255)];
		for (p in 0...16)
			for (c in 0...4)
				check(pixels.get(p * 4 + c) == expected[c], 'pixel $p channel $c is ${pixels.get(p * 4 + c)}, not ${expected[c]}');
		Sys.println('render: 4x4 pixels of ${[for (c in 0...4) pixels.get(c)].join(", ")}');
		blended(device, queue);
	}

	// The check draws a half-transparent red quad over a blue target. The result is half
	// red, half blue and opaque.
	static function blended(device:Device, queue:CommandQueue) {
		var pixels = quad(device, queue, [1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0], ALPHA);
		var expected = [128, 0, 128, 255];
		for (p in 0...16)
			for (c in 0...4)
				check(Math.abs(pixels.get(p * 4 + c) - expected[c]) <= 1, 'blended pixel $p channel $c is ${pixels.get(p * 4 + c)}, not about ${expected[c]}');
		Sys.println('blend: 4x4 pixels of about ${expected.join(", ")}');
	}

	// Draws a quad of `tint` into a 4x4 target cleared to `clear`, with `blend`, and
	// returns the target's pixels.
	static function quad(device:Device, queue:CommandQueue, tint:Array<Float>, clear:Array<Float>, blend:Blend):Bytes {
		// The metallib the build made, or the MSL compiled now where it had no Metal compiler.
		var library = hxn.gpu.metal.Shaders.library(device, Tinted.metallib(), Tinted.MSL);
		var vertex = library.functionNamed("vertex_main");
		var fragment = library.functionNamed("fragment_main");
		// The quad's positions sit in a vertex buffer at an index no shader resource uses.
		var vertices = 30;
		var pipeline = device.newRenderPipelineAsync(vertex, fragment, PixelFormat.Rgba8Unorm,
			[{location: Tinted.INPUT_position, format: VertexFormat.Float2, offset: 0, buffer: vertices}],
			[{buffer: vertices, stride: 8, perInstance: false}], blend).await();
		var size = 4;
		var target = device.newTexture({
			width: size,
			height: size,
			format: PixelFormat.Rgba8Unorm,
			usage: TextureUsage.RenderTarget | TextureUsage.ShaderRead,
			storage: StorageMode.Shared
		});

		var quad = Bytes.alloc(4 * 8);
		var corners = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
		for (i in 0...corners.length)
			quad.setFloat(i * 4, corners[i]);
		var positions = device.newBufferWith(quad);
		var params = Bytes.alloc(Tinted.PARAMS_SIZE);
		for (i in 0...4)
			params.setFloat(Tinted.PARAMS_tint + i * 4, tint[i]);

		var commands = queue.commandBuffer();
		var encoder = commands.renderEncoder(target, LoadAction.Clear, clear[0], clear[1], clear[2], clear[3]);
		encoder.setPipeline(pipeline);
		encoder.setVertexBuffer(positions, 0, vertices);
		encoder.setFragmentBytes(params, slot(Tinted.MSL_SLOTS, "fragment_main", Tinted.PARAMS_GROUP, Tinted.PARAMS_BINDING));
		encoder.draw(Primitive.TriangleStrip, 0, 4, 1);
		encoder.end();
		commands.commit();
		commands.waitUntilCompleted();

		var pixels = target.read(size * 4);
		closeAll([encoder, commands, positions, target, pipeline, fragment, vertex, library]);
		return pixels;
	}
}
