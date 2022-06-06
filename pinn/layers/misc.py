# -*- coding: utf-8 -*-

"""Misc. (Keras) Layers for Atomistic Neural Networks"""

import tensorflow as tf

class AtomicOnehot(tf.keras.layers.Layer):
    """ One-hot encoding Lyaer

    perform one-hot encoding for elements
    """
    def __init__(self, atom_types=[1, 6, 7, 8, 9]):
        super(AtomicOnehot, self).__init__()
        self.atom_types = atom_types

    def call(self, elems):
        output = tf.equal(tf.expand_dims(elems, 1),
                          tf.expand_dims(self.atom_types, 0))
        return output

class ANNOutput(tf.keras.layers.Layer):
    """ ANN Ouput layer

    Output atomic or molecular (system) properties
    """
    def __init__(self, out_pool):
        super(ANNOutput, self).__init__()
        self.out_pool = out_pool

    def call(self, tensors):
        ind_1, output = tensors

        if self.out_pool:
            out_pool = {'sum': tf.math.unsorted_segment_sum,
                        'max': tf.math.unsorted_segment_max,
                        'min': tf.math.unsorted_segment_min,
                        'avg': tf.math.unsorted_segment_mean,
            }[self.out_pool]
            output =  out_pool(output, ind_1[:,0],
                               tf.reduce_max(ind_1)+1)
        output = tf.squeeze(output, axis=1)

        return output


class DensityEstimate(tf.keras.layers.Layer):
    """ Density Estimate Layer"""
    def __init__(self, rc, cutoff_type, ddrb_ref):
        super(DensityEstimate, self).__init__()
        r = np.linspace(0, rc, 100)
        if cutoff_type=='f1':
            fac = np.trapz((0.5*np.cos(np.pi*r/rc)+1)*(4*np.pi*r**2), r)
        elif cutoff_type=='f2':
            r = np.linspace(0, rc, 100)
            fac = np.trapz((np.tanh(1-r/rc)/np.tanh(1))**3*(4*np.pi*r**2), r)
        else:
            raise NotImplementedError(f'Unknown Cutoff {cutoff_type}')
        self.cnt_ref = ddrb_ref*fac


    def call(self, ind_2, fc, pairwise=False):
        """
        Args:
           ind_2: [N_pair, 2] indices of each pair
           fc: [N_pair] cutoff function evaluated on each pair
        Returns:
           density estimation
        """
        cnt = tf.
        rho = cnt/self.cnt_ref
        if pairwise:
            return tf.gather(rho, ind_2[:,0])
        else:
            return rho
