"""Parameter counts must reproduce the manuscript (Table 2, Section 4.6)."""
import numpy as np
import tensorflow as tf

from talstmae import baselines as B
from talstmae import models as M


def test_hblt_parameter_counts_match_manuscript():
    df = M.parameter_check()
    bad = df[df.check == "MISMATCH"]
    assert bad.empty, bad.to_string()


def test_tae_layer_counts_match_table2():
    for d, lat in [(122, 16), (194, 23), (78, 16)]:
        ae, _ = M.build_tae(d, lat)
        p = dict(M.param_table(ae).values)
        assert p["enc_mha"] == 1088 and p["enc_layernorm"] == 32 and p["enc_bigru"] == 3264
        assert p["dec_bigru"] == 9600 and p["dec_mha"] == 8352
        assert p["latent"] == 32 * lat + lat
        assert p["enc_dense256"] + p["enc_bn"] == d * 256 + 256 + 1024
        assert p["dec_dense"] == 256 * d + d


def test_deformable_conv_shapes_and_gradients():
    layer = B.DeformableConv1D(8, 3)
    x = tf.random.normal((4, 20, 1))
    with tf.GradientTape() as tape:
        y = layer(x)
        loss = tf.reduce_sum(y ** 2)
    assert y.shape == (4, 20, 8)
    grads = tape.gradient(loss, layer.trainable_variables)
    assert all(g is not None for g in grads)


def test_deformable_conv_equals_standard_conv_at_zero_offset():
    layer = B.DeformableConv1D(4, 3)
    x = tf.random.normal((2, 12, 3))
    y = layer(x).numpy()
    conv = tf.keras.layers.Conv1D(4, 3, padding="same")
    conv.build(x.shape)
    conv.set_weights([layer.kernel.numpy(), layer.bias.numpy()])
    ref = conv(x).numpy()
    # interior positions are identical (borders differ: clipping vs zero padding)
    np.testing.assert_allclose(y[:, 1:-1], ref[:, 1:-1], atol=1e-5)


def test_baselines_build():
    for d in (122, 194, 78):
        assert B.build_adfcnn_bilstm(d).count_params() > 0
        ae, enc = B.build_hae(d, 16)
        assert enc.output_shape[-1] == 16
        assert B.build_hrl(16).count_params() > 0
