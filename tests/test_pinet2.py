from pinn.networks.pinet2 import DotLayer, ScaleLayer, PIXLayer
import pytest
import tensorflow as tf
import numpy as np


def create_rot_mat(theta):
    return tf.constant([
        [1., 0., 0.],
        [0., np.cos(theta), -np.sin(theta)],
        [0., np.sin(theta), np.cos(theta)]
    ], dtype=tf.float32)


class TestPiNet2:

    @pytest.mark.forked
    def test_simple_dotlayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5

        prop = tf.random.uniform((nsamples, ndims, nchannels))
        # create a rotation matrix
        theta = 42.
        rot = create_rot_mat(theta)
        rot = tf.constant(rot, dtype=tf.float32)

        dot = DotLayer('simple')
        tf.debugging.assert_near(
            dot(prop), dot(tf.einsum('ixa,xy->iya', prop, rot))
        )

    @pytest.mark.forked
    def test_general_dotlayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5

        prop = tf.random.uniform((nsamples, ndims, nchannels))
        theta = 42.
        rot = create_rot_mat(theta)

        dot = DotLayer('general')
        tf.debugging.assert_near(
            dot(prop), dot(tf.einsum('ixa,xy->iya', prop, rot))
        )

    @pytest.mark.forked
    def test_scalelayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5

        px = tf.random.uniform((nsamples, ndims, nchannels))
        p1 = tf.random.uniform((nsamples, nchannels))

        rot = create_rot_mat(42.)

        scaler = ScaleLayer()
        out = scaler([px, p1])
        assert out.shape == (nsamples, ndims, nchannels)
        tf.debugging.assert_near(
            tf.einsum('ixa,xy->iya', scaler([px, p1]), rot), scaler([tf.einsum('ixa,xy->iya', px, rot), p1])
        )

    @pytest.mark.forked
    def test_simple_pixlayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5
        nnbors = 3
        px = tf.random.uniform((nsamples, ndims, nchannels))
        ind_2 = tf.random.uniform((nnbors, 2), maxval=nsamples, dtype=tf.int32)
        rot = create_rot_mat(42.)

        pix = PIXLayer('simple')
        out = pix([ind_2, px])
        assert out.shape == (nnbors, ndims, nchannels)
        tf.debugging.assert_near(
            tf.einsum('ixa,xy->iya', pix([ind_2, px]), rot), pix([ind_2, tf.einsum('ixa,xy->iya', px, rot)])
        )


    @pytest.mark.forked
    def test_general_pixlayer(self):

        nsamples = 10
        ndims = 3
        nchannels = 5
        nnbors = 3
        px = tf.random.uniform((nsamples, ndims, nchannels))
        ind_2 = tf.random.uniform((nnbors, 2), maxval=nsamples, dtype=tf.int32)
        rot = create_rot_mat(42.)

        pix = PIXLayer('general')
        out = pix([ind_2, px])
        assert out.shape == (nnbors, ndims, nchannels)
        tf.debugging.assert_near(
            tf.einsum('ixa,xy->iya', pix([ind_2, px]), rot), pix([ind_2, tf.einsum('ixa,xy->iya', px, rot)])
        )
