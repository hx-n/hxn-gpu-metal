package gpu.metal;

#if hxnsl
/**
	An hxnsl shader for Metal. It gets hxnsl's `MSL` and `MSL_SLOTS`, and
	`metallib()`: the shader compiled to a Metal library during the build,
	or null where the build machine has no Metal compiler.
	`Shaders.library` loads whichever it has.
**/
interface Shader extends hxnsl.Shader {}
#end
