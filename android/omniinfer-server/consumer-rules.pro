# JNI_OnLoad in omniinfer_jni.cpp registers this bridge by hard-coded class and
# method names. Release shrinking/obfuscation must preserve the bridge type and
# its native method signatures, or RegisterNatives will fail at library load.
-keep class com.omniinfer.server.OmniInferBridge { *; }
-keep class com.omniinfer.server.OmniInferServer { *; }
-keep class com.omniinfer.server.OmniInferStreamCallback { *; }

# LiteRtLmBackendFactory is accessed via reflection in LiteRtLmBackendSupport
# (Class.forName + getMethod). Preserve class and method names so the
# reflective create() call works after R8 obfuscation.
-keep class com.omniinfer.server.LiteRtLmBackendFactory { *; }

# LiteRT-LM native code calls these callback methods by exact Java names.
# Without these rules, R8 can rename the generated callback implementation to
# something like Lzd/a; and nativeSendMessageAsync aborts when it looks up
# onMessage(String), onDone(), or onError(int, String).
-keep class com.google.ai.edge.litertlm.LiteRtLmJni { *; }
-keep class com.google.ai.edge.litertlm.LiteRtLmJni$JniMessageCallback { *; }
-keep class com.google.ai.edge.litertlm.LiteRtLmJni$JniInferenceCallback { *; }
-keep class com.google.ai.edge.litertlm.Conversation$JniMessageCallbackImpl { *; }
-keep class com.google.ai.edge.litertlm.Session$JniInferenceCallbackImpl { *; }

# Conversation.getBenchmarkInfo() constructs BenchmarkInfo from native code.
# Preserve its Java name and constructor for apps that opt in to LiteRT-LM
# benchmark metrics; otherwise R8 can remove the constructor and ART aborts
# with "JNI DETECTED ERROR IN APPLICATION: mid == null".
-keep class com.google.ai.edge.litertlm.BenchmarkInfo { *; }
