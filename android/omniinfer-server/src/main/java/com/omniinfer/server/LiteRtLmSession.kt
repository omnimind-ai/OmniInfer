package com.omniinfer.server

interface LiteRtLmSession {
    fun generate(
        messagesJson: String,
        imageDataArray: Array<ByteArray>?,
        requestJson: String,
        callback: OmniInferStreamCallback?,
    ): String

    fun cancel()
    fun reset()
    fun close()
    fun diagnostics(): Map<String, String>
}
