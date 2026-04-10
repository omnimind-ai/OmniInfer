package com.omniinfer.server

import androidx.annotation.Keep

/**
 * JNI looks up these callback methods by hard-coded names at runtime.
 * Keep them on a stable concrete class so release shrinking cannot rename them.
 */
@Keep
class OmniInferStreamCallback(
    private val onTokenHandler: ((String) -> Unit)? = null,
    private val onMetricsHandler: ((String) -> Unit)? = null,
) {
    @Keep
    fun onToken(token: String) {
        onTokenHandler?.invoke(token)
    }

    @Keep
    fun onMetrics(metrics: String) {
        onMetricsHandler?.invoke(metrics)
    }
}
