package hxn.gpu.metal;

/** Shaders from hxnsl, for Metal. **/
class Shaders {
	#if macro
	/**
		Registers the Metal targets for `hxn.gpu.metal.Shader`. The package's
		extraParams.hxml runs it; without hxnsl it does nothing.
	**/
	public static function register():Void {
		#if hxnsl
		hxnsl.Targets.register(new hxn.gpu.metal.MetalLibrary(), "hxn.gpu.metal.Shader");
		hxnsl.Targets.register(new hxnsl.targets.Msl(), "hxn.gpu.metal.Shader");
		#end
	}
	#else

	/**
		The shader's library: its `metallib()` when one was compiled with the program, so
		nothing is compiled now, else its `MSL`, compiled now.
	**/
	public static function library(device:Device, metallib:Null<haxe.io.Bytes>, msl:String):Library {
		return metallib != null ? device.newLibrary(metallib) : device.newLibraryFromSource(msl);
	}
	#end
}
