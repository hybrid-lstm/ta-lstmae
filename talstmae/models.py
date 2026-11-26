"""TA-LSTMAE models: Temporal Attention Autoencoder (TAE) and the HBLT classifier family.

Parameter counts reproduce the manuscript exactly (see tests/test_models.py):
Table 2 (TAE) and Section 4.6 (CNN only 28,833; Transformer only 81,089;
BiLSTM only 104,001; LSTM-unidirectional fusion 128,033; full HBLT 179,617;
No Feature Selection 261,025 / 227,233).

Implementation notes:
* The decoder input projection Dense(64)+Reshape(4x16), which is not listed in
  Table 2, gives the decoder BiGRU the 16-d input that its 9,600 parameters
  imply on every dataset.
* The gated-fusion head and the BatchNorm in the CNN branch are implementation
  details that Section 3.3 does not spell out; all parameter counts match the
  manuscript.


"""
from __future__ import annotations

import keras
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers as L


# ----------------------------------------------------------------------------- TAE (Section 3.2, Table 2)
def build_tae(input_dim: int, latent_dim: int, lr: float = 1e-3):
    inp = L.Input((input_dim,), name="tae_input")
    x = L.Dense(256, activation="relu", name="enc_dense256")(inp)
    x = L.BatchNormalization(name="enc_bn")(x)
    x = L.Reshape((16, 16), name="enc_reshape")(x)
    a = L.MultiHeadAttention(num_heads=4, key_dim=4, name="enc_mha")(x, x)
    x = L.LayerNormalization(name="enc_layernorm")(L.Add(name="enc_add")([x, a]))
    x = L.Bidirectional(L.GRU(16), name="enc_bigru")(x)                          # [B, 32]
    z = L.Dense(latent_dim, name="latent")(x)                                    # [B, d]
    d = L.Dense(64, activation="relu", name="dec_project")(z)
    d = L.Reshape((4, 16), name="dec_reshape")(d)
    d = L.Bidirectional(L.GRU(32, return_sequences=True), name="dec_bigru")(d)   # [B, 4, 64]
    d = L.MultiHeadAttention(num_heads=4, key_dim=8, name="dec_mha")(d, d)
    d = L.Flatten(name="dec_flatten")(d)
    out = L.Dense(input_dim, activation="sigmoid", name="dec_dense")(d)
    ae = Model(inp, out, name="TAE")
    enc = Model(inp, z, name="TAE_encoder")
    ae.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="mse")
    return ae, enc


# ----------------------------------------------------------------------------- HBLT (Section 3.3)
@keras.saving.register_keras_serializable(package="talstmae")
class GatedMix(L.Layer):
    """Gated fusion g*h + (1-g)*s (parameter-free; gate and candidate are separate Dense layers)."""

    def call(self, inputs):
        g, h, s = inputs
        return g * h + (1.0 - g) * s


def transformer_block(x, ff_dim: int, name: str, dropout: float):
    a = L.MultiHeadAttention(num_heads=4, key_dim=16, name=f"{name}_mha")(x, x)
    x = L.LayerNormalization(name=f"{name}_ln1")(L.Add()([x, L.Dropout(dropout)(a)]))
    f = L.Dense(ff_dim, activation="relu", name=f"{name}_ff1")(x)
    f = L.Dense(64, name=f"{name}_ff2")(f)
    return L.LayerNormalization(name=f"{name}_ln2")(L.Add()([x, L.Dropout(dropout)(f)]))


CONFIGS = {  # Table 6 rows + the genuinely causal model of Section 4.6
    "No Feature Selection + HBLT": dict(branches=("bilstm", "transformer", "cnn"), raw_input=True),
    "TAE + LSTM (unidirectional)": dict(branches=("lstm", "transformer", "cnn")),
    "TAE + CNN only": dict(branches=("cnn",)),
    "TAE + BiLSTM only": dict(branches=("bilstm",)),
    "TAE + Transformer only": dict(branches=("transformer",)),
    "TAE + HBLT (Full Proposed)": dict(branches=("bilstm", "transformer", "cnn")),
    "Causal-only (unidirectional LSTM)": dict(branches=("lstm",)),
}
FULL = "TAE + HBLT (Full Proposed)"
CAUSAL = "Causal-only (unidirectional LSTM)"


def build_hblt(seq_len: int, input_dim: int, branches=("bilstm", "transformer", "cnn"), dropout: float = 0.2,
               lr: float = 1e-3, name: str = "HBLT"):
    inp = L.Input((seq_len, input_dim), name="window_input")
    outs = []
    if "bilstm" in branches:
        b = L.Bidirectional(L.LSTM(64, return_sequences=True, dropout=dropout), name="bilstm_1")(inp)
        b = L.Bidirectional(L.LSTM(32, return_sequences=True, dropout=dropout), name="bilstm_2")(b)
        outs.append(L.Dense(64, name="bilstm_proj")(b))
    if "lstm" in branches:                                                       # causal recurrent branch
        b = L.LSTM(64, return_sequences=True, dropout=dropout, name="lstm_1")(inp)
        b = L.LSTM(32, return_sequences=True, dropout=dropout, name="lstm_2")(b)
        outs.append(L.Dense(64, name="lstm_proj")(b))
    if "transformer" in branches:
        t = L.Dense(64, name="tr_embed")(inp)
        t = transformer_block(t, 128, "tr_block1", dropout)
        t = transformer_block(t, 64, "tr_block2", dropout)
        outs.append(L.Dense(64, name="tr_proj")(t))
    if "cnn" in branches:
        c = L.Conv1D(64, 3, padding="same", activation="relu", name="cnn_conv1")(inp)
        c = L.BatchNormalization(name="cnn_bn")(c)
        c = L.Conv1D(32, 3, padding="same", activation="relu", name="cnn_conv2")(c)
        outs.append(L.Dense(64, name="cnn_proj")(c))
    s = outs[0] if len(outs) == 1 else L.Add(name="fusion_add")(outs)
    g = L.Dense(64, activation="sigmoid", name="fusion_gate")(s)
    h = L.Dense(64, activation="tanh", name="fusion_candidate")(s)
    f = GatedMix(name="gated_fusion")([g, h, s])
    f = L.LayerNormalization(name="fusion_ln")(f)
    f = L.GlobalAveragePooling1D(name="gap")(f)                                 # 64-d joint representation
    f = L.BatchNormalization(name="head_bn")(f)
    f = L.Dense(128, activation="relu", name="dense128")(f)
    f = L.Dropout(dropout)(f)
    out = L.Dense(1, activation="sigmoid", name="output")(f)
    m = Model(inp, out, name=name)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy",
              metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")])
    return m


REPORTED_PARAMS = {("TAE + CNN only", 16): 28_833, ("TAE + Transformer only", 16): 81_089,
                   ("TAE + BiLSTM only", 16): 104_001, ("TAE + LSTM (unidirectional)", 16): 128_033,
                   (FULL, 16): 179_617, ("No Feature Selection + HBLT", 122): 261_025,
                   ("No Feature Selection + HBLT", 78): 227_233}


def parameter_check(seq_len: int = 10) -> pd.DataFrame:
    rows = []
    for (cfg_name, d), expected in REPORTED_PARAMS.items():
        n = build_hblt(seq_len, d, CONFIGS[cfg_name]["branches"]).count_params()
        rows.append((cfg_name, d, n, expected, "OK" if n == expected else "MISMATCH"))
    rows.append((CAUSAL, 16, build_hblt(seq_len, 16, CONFIGS[CAUSAL]["branches"]).count_params(), None, "—"))
    return pd.DataFrame(rows, columns=["configuration", "input_dim", "params", "reported", "check"])


def param_table(model) -> pd.DataFrame:
    return pd.DataFrame([(l.name, l.count_params()) for l in model.layers if l.count_params() > 0],
                        columns=["layer", "params"])
