package hxn.gpu.metal;

#if (macro && hxnsl)
import haxe.io.Bytes;
import haxe.macro.Context;
import haxe.macro.Expr;
import sys.io.Process;

/**
	An hxnsl target that compiles a shader's MSL to a Metal library at
	compile time, with `xcrun metal` and `metallib`, for the SDK the program
	targets: iOS (`-D iphoneos`), the iOS simulator (`-D iphonesim`) or
	macOS. Without a Metal compiler on the compiling machine the shader's
	`metallib()` is null, and the program compiles its MSL when it loads it.
**/
class MetalLibrary extends hxnsl.Target {
	static var versions = new Map<String, Null<String>>();
	static var warned = false;

	public function new() {
		super("metallib");
	}

	static function sdk():String {
		return if (Context.defined("iphonesim")) "iphonesimulator" else if (Context.defined("iphoneos")) "iphoneos" else "macosx";
	}

	/** The Metal compiler's version line for `sdk`, or null without one. **/
	static function version(sdk:String):Null<String> {
		if (versions.exists(sdk))
			return versions.get(sdk);
		var found:Null<String> = null;
		if (Sys.systemName() == "Mac") {
			var p = new Process("xcrun", ["-sdk", sdk, "metal", "--version"]);
			var out = p.stdout.readAll().toString();
			p.stderr.readAll();
			if (p.exitCode() == 0)
				found = StringTools.trim(out.split("\n")[0]);
			p.close();
		}
		versions.set(sdk, found);
		return found;
	}

	static function xcrun(shader:hxnsl.Compiled, arguments:Array<String>) {
		var p = new Process("xcrun", arguments);
		p.stdout.readAll();
		var errors = p.stderr.readAll().toString();
		var code = p.exitCode();
		p.close();
		if (code != 0)
			shader.error('xcrun ${arguments.join(" ")} failed: $errors');
	}

	override public function build(s:hxnsl.Compiled):Array<Field> {
		var sdk = sdk();
		var compiler = version(sdk);
		var value:Expr = macro null;
		if (compiler != null) {
			var standard = (sdk == "macosx" ? "macos" : "ios") + "-metal2.1";
			var out = s.step('$sdk $standard $compiler', ["shader.metal" => Bytes.ofString(s.msl)], dir -> {
				xcrun(s, ["-sdk", sdk, "metal", '-std=$standard', "-c", '$dir/shader.metal', "-o", '$dir/shader.air']);
				xcrun(s, ["-sdk", sdk, "metallib", '$dir/shader.air', "-o", '$dir/shader.metallib']);
			}, ["shader.metallib"]);
			var resource = s.resource("shader.metallib", out.get("shader.metallib"));
			value = macro haxe.Resource.getBytes($v{resource});
		} else if (!warned) {
			warned = true;
			Context.warning('hxn-gpu-metal: no Metal compiler for $sdk (xcrun -sdk $sdk metal); shaders are compiled from MSL when the program loads them',
				s.pos);
		}
		return [{
			name: "metallib",
			doc: "The shader as a Metal library built for this compile's SDK, or null where no Metal compiler was found at compile time.",
			access: [APublic, AStatic],
			kind: FFun({args: [], ret: macro :Null<haxe.io.Bytes>, expr: macro return $value}),
			pos: s.pos
		}];
	}
}
#end
