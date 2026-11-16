package hxn.gpu.metal;

#if hxnsl
/**
	An hxnsl shader for Metal. It gets hxnsl's `MSL` and `MSL_SLOTS`, and
	`metallib()`: the shader compiled to a Metal library at compile time,
	or null where the compiling machine has no Metal compiler.
	`Shaders.library` loads whichever it has.
**/
interface Shader extends hxnsl.Shader {}
#end
