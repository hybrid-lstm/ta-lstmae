"""Baselines re-implemented for the controlled head-to-head comparison.

Both baselines are trained and evaluated under the SAME protocol as TA-LSTMAE:
identical official/chronological partitions, contiguous validation block,
one-hot + Min-Max preprocessing fitted on the training split, class-weighted
loss (no resampling), the same test targets (records with index >= seq_len-1,
i.e. exactly the final events of the TA-LSTMAE test windows), the same
decision threshold, and the same metrics engine (including MTTD).

Both models are record-level classifiers, as in their original papers; the
temporal context available to TA-LSTMAE (the nine preceding events) is NOT
given to them. Hyperparameters that the original papers do not report are
listed in BASELINE_NOTES and must be stated as such in the manuscript.
"""
from __future__ import annotations

import keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as L

BASELINE_NOTES = {
    "HAE-HRL (re-implemented)": (
        "Xue, Kang and Yu, Comput. Secur. 151 (2025) 104328. Re-implemented from the published description: "
        "HAE = CNN-GRU autoencoder with dual (max + average) pooling, latent size matched to TA-LSTMAE "
        "(16/23/16); HRL = 1D-CNN stem + two residual blocks + LSTM classifier. Filter counts (32/64), "
        "LSTM units (64), Dense(64) head and dropout 0.2 are assumptions where the paper is not explicit. "
        "SMOTE+RUS of the original are replaced by class weighting (common protocol)."),
    "ADFCNN-BiLSTM (re-implemented)": (
        "Li, Li and Jia, Sensors 25(5) (2025) 1382. Two blocks of 1D deformable convolution (128 and 256 "
        "channels, kernel 3) + ReLU + MaxPool(4) + BatchNorm + ECA, spatial attention module, BiLSTM + "
        "multi-head attention, dropout and fully connected head; Adam, lr 5e-4 (NSL-KDD, UNSW-NB15) or 1e-3. "
        "BiLSTM units (64), attention heads (4), SAM kernel (7), dropout (0.3) and Dense(64) are assumptions "
        "(not reported). SMOTE-ENN + EQLv2 loss of the original are replaced by class weighting (common protocol)."),
}


# ----------------------------------------------------------------------------- HAE-HRL
def build_hae(input_dim: int, latent_dim: int, lr: float = 1e-3):
    inp = L.Input((input_dim,), name="hae_input")
    x = L.Reshape((input_dim, 1))(inp)
    x = L.Conv1D(32, 3, padding="same", activation="relu", name="hae_conv1")(x)
    x = L.Concatenate(name="hae_dual_pool")([L.MaxPooling1D(2, padding="same")(x),
                                             L.AveragePooling1D(2, padding="same")(x)])
    x = L.Conv1D(64, 3, padding="same", activation="relu", name="hae_conv2")(x)
    x = L.GRU(64, name="hae_gru")(x)
    z = L.Dense(latent_dim, name="hae_latent")(x)
    d = L.Dense(64, activation="relu", name="hae_dec1")(z)
    out = L.Dense(input_dim, activation="sigmoid", name="hae_dec_out")(d)
    ae, enc = Model(inp, out, name="HAE"), Model(inp, z, name="HAE_encoder")
    ae.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return ae, enc


def _res_block(x, filters: int, name: str):
    y = L.Conv1D(filters, 3, padding="same", name=f"{name}_c1")(x)
    y = L.Activation("relu")(L.BatchNormalization(name=f"{name}_bn1")(y))
    y = L.Conv1D(filters, 3, padding="same", name=f"{name}_c2")(y)
    y = L.BatchNormalization(name=f"{name}_bn2")(y)
    return L.Activation("relu")(L.Add(name=f"{name}_add")([x, y]))


def build_hrl(latent_dim: int, dropout: float = 0.2, lr: float = 1e-3):
    inp = L.Input((latent_dim,), name="hrl_input")
    x = L.Reshape((latent_dim, 1))(inp)
    x = L.Conv1D(64, 3, padding="same", activation="relu", name="hrl_stem")(x)
    x = _res_block(x, 64, "hrl_res1")
    x = _res_block(x, 64, "hrl_res2")
    x = L.LSTM(64, name="hrl_lstm")(x)
    x = L.Dense(64, activation="relu", name="hrl_dense")(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation="sigmoid", name="hrl_out")(x)
    m = Model(inp, out, name="HRL")
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    return m


# ----------------------------------------------------------------------------- ADFCNN-BiLSTM
@keras.saving.register_keras_serializable(package="talstmae")
class DeformableConv1D(L.Layer):
    """1D deformable convolution (DCNv1-style): learned fractional offsets per position and kernel tap,
    linear interpolation of the sampled inputs, then a standard convolution over the sampled taps."""

    def __init__(self, filters: int, kernel_size: int = 3, **kw):
        super().__init__(**kw)
        self.filters, self.kernel_size = filters, kernel_size

    def build(self, input_shape):
        c_in = int(input_shape[-1])
        k = self.kernel_size
        self.offset_conv = L.Conv1D(k, k, padding="same", kernel_initializer="zeros", bias_initializer="zeros",
                                    name="offsets")
        self.offset_conv.build(input_shape)
        self.kernel = self.add_weight(name="kernel", shape=(k, c_in, self.filters), initializer="glorot_uniform")
        self.bias = self.add_weight(name="bias", shape=(self.filters,), initializer="zeros")

    def call(self, x):
        k = self.kernel_size
        length = tf.shape(x)[1]
        lf = tf.cast(length, x.dtype)
        offsets = self.offset_conv(x)                                                      # [B, L, K]
        base = (tf.cast(tf.range(length), x.dtype)[None, :, None]
                + tf.cast(tf.range(k) - k // 2, x.dtype)[None, None, :])                  # [1, L, K]
        pos = tf.clip_by_value(base + offsets, 0.0, lf - 1.0)
        i0 = tf.floor(pos)
        w1 = pos - i0
        i0 = tf.cast(i0, tf.int32)
        i1 = tf.minimum(i0 + 1, length - 1)
        b = tf.shape(x)[0]
        g0 = tf.gather(x, tf.reshape(i0, [b, -1]), batch_dims=1)                          # [B, L*K, C]
        g1 = tf.gather(x, tf.reshape(i1, [b, -1]), batch_dims=1)
        c = tf.shape(x)[2]
        g0 = tf.reshape(g0, [b, length, k, c])
        g1 = tf.reshape(g1, [b, length, k, c])
        sampled = (1.0 - w1)[..., None] * g0 + w1[..., None] * g1                          # [B, L, K, C]
        return tf.einsum("blkc,kcf->blf", sampled, self.kernel) + self.bias

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.filters)

    def get_config(self):
        return {**super().get_config(), "filters": self.filters, "kernel_size": self.kernel_size}


@keras.saving.register_keras_serializable(package="talstmae")
class ECA(L.Layer):
    """Efficient Channel Attention: 1D convolution across pooled channel descriptors."""

    def __init__(self, k: int = 3, **kw):
        super().__init__(**kw)
        self.k = k
        self.conv = L.Conv1D(1, k, padding="same", use_bias=False)

    def call(self, x):
        s = tf.reduce_mean(x, axis=1)[..., None]                    # [B, C, 1]
        w = tf.sigmoid(self.conv(s))                                # [B, C, 1]
        return x * tf.transpose(w, [0, 2, 1])

    def get_config(self):
        return {**super().get_config(), "k": self.k}


@keras.saving.register_keras_serializable(package="talstmae")
class SpatialAttention1D(L.Layer):
    """Spatial attention module (CBAM-style) along the sequence axis."""

    def __init__(self, k: int = 7, **kw):
        super().__init__(**kw)
        self.k = k
        self.conv = L.Conv1D(1, k, padding="same", activation="sigmoid")

    def call(self, x):
        a = tf.concat([tf.reduce_mean(x, -1, keepdims=True), tf.reduce_max(x, -1, keepdims=True)], -1)
        return x * self.conv(a)

    def get_config(self):
        return {**super().get_config(), "k": self.k}


def build_adfcnn_bilstm(input_dim: int, lr: float = 5e-4, dropout: float = 0.3):
    inp = L.Input((input_dim,), name="adf_input")
    x = L.Reshape((input_dim, 1))(inp)
    for i, ch in enumerate((128, 256), 1):
        x = DeformableConv1D(ch, 3, name=f"adf_dconv{i}")(x)
        x = L.Activation("relu")(x)
        x = L.MaxPooling1D(4, padding="same", name=f"adf_pool{i}")(x)
        x = L.BatchNormalization(name=f"adf_bn{i}")(x)
        x = ECA(3, name=f"adf_eca{i}")(x)
    x = SpatialAttention1D(7, name="adf_sam")(x)
    x = L.Bidirectional(L.LSTM(64, return_sequences=True), name="adf_bilstm")(x)
    a = L.MultiHeadAttention(num_heads=4, key_dim=32, name="adf_mha")(x, x)
    x = L.LayerNormalization(name="adf_ln")(L.Add()([x, a]))
    x = L.GlobalAveragePooling1D()(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(64, activation="relu", name="adf_fc")(x)
    out = L.Dense(1, activation="sigmoid", name="adf_out")(x)
    m = Model(inp, out, name="ADFCNN_BiLSTM")
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    return m


ADFCNN_LR = {"nsl_kdd": 5e-4, "unsw_nb15": 5e-4, "cicids2018": 1e-3}
