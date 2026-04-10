# JNI_OnLoad in omniinfer_jni.cpp registers this bridge by hard-coded class and
# method names. Release shrinking/obfuscation must preserve the bridge type and
# its native method signatures, or RegisterNatives will fail at library load.
-keep class com.omniinfer.server.OmniInferBridge { *; }
-keep class com.omniinfer.server.OmniInferServer { *; }
-keep class com.omniinfer.server.OmniInferStreamCallback { *; }
