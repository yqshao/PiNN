# -*- coding: utf-8 -*-
#
# Implementation of GLE optimizer, to use with the Esitmator API

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers.legacy import Optimizer

tfp_cholesky = tfp.experimental.linalg.simple_robustified_cholesky

# Preset drift matrix parameterizations from GLEOPT, with fixed omega_0=1
# fmt: off
_A_PRESETS = {
    # Classical Langevin Equation
    'LE': [[1]],
    # http://gle4md.org/index.html?page=matrix&kind=optimal&tgt=kv&range=2-2&w0=1&uw0=auw&outmode=python&aunits=aut
    'OS-NS2-R2': np.array([
        [   3.661845071343e+0,   -5.159467179216e-1,    4.183948401978e+0, ],
        [   9.154985899852e-1,    7.636263702277e-2,    8.915201434235e-1, ],
        [   2.492071207845e+0,   -8.915201434235e-1,    3.624125365875e+0, ],
    ]),
    # http://gle4md.org/index.html?page=matrix&kind=optimal&tgt=kv&range=4-4&w0=1&uw0=auw&outmode=python&aunits=aut
    'OS-NS4-R4': np.array([
        [   2.468046446184e+1,    3.618484092955e-2,    1.529754814420e+0,   -4.832976827822e+0,    3.075592075613e+1, ],
        [  -3.690906085933e-2,    1.140757551908e-5,    9.580997856843e-2,   -2.633785790846e-2,    5.628596264598e-2, ],
        [  -1.967695098242e+0,   -9.580997856843e-2,    1.803797219554e-1,    6.834981599580e-1,   -1.326536023287e+0, ],
        [  -1.376606625580e+0,    2.633785790846e-2,   -6.834981599580e-1,    3.538593708082e+0,    1.527314745454e+0, ],
        [   2.893495045182e+1,   -5.628596264598e-2,    1.326536023287e+0,   -1.527314745454e+0,    4.108827033038e+1, ],
    ]),
    # http://gle4md.org/index.html?page=matrix&kind=optimal&tgt=kv&range=6-6&w0=1&uw0=auw&outmode=python&aunits=aut
    'OS-NS6-R6': np.array([
        [   2.003451461738e+2,    1.195166532140e-2,   -1.501589105817e-1,    7.780288272245e-1,   -4.715415455481e+0,   -3.369426395468e+1,    3.087879262440e+2, ],
        [   2.987786172852e-3,    2.973003949060e-3,    2.164783639462e-3,    8.389689749648e-3,    3.718433690545e-2,   -5.599950848003e-2,   -2.703236308866e-2, ],
        [  -8.315118930700e-3,   -2.164783639462e-3,    5.545886327765e-2,   -4.995234488533e-2,    4.538281042979e-2,   -2.887861541262e-1,   -3.910356363837e-2, ],
        [   6.366393897191e-1,   -8.389689749648e-3,    4.995234488533e-2,    7.279035700316e-1,   -6.832793153611e-4,    7.994393522225e-2,    1.218045274817e-1, ],
        [  -5.027237833082e+0,   -3.718433690545e-2,   -4.538281042979e-2,    6.832793153611e-4,    6.566615364366e+0,    5.397941970852e-2,   -1.635309543931e-1, ],
        [  -3.392472744143e+1,    5.599950848003e-2,    2.887861541262e-1,   -7.994393522225e-2,   -5.397941970852e-2,    5.579962068285e+1,    2.277766277868e-1, ],
        [   3.075626051613e+2,    2.703236308866e-2,    3.910356363837e-2,   -1.218045274817e-1,    1.635309543931e-1,   -2.277766277868e-1,    5.413899128627e+2, ],
    ]),
}
# fmt: on


def _thermostat_step(s, p, xi, T, S, sm):
    st1 = s[0].assign(p / sm)
    with tf.control_dependencies([st1]):
        st2 = s.assign(
            tf.einsum("ij,j...->i...", T, s) + tf.einsum("...ij,j...->i...", S, xi)
        )
    with tf.control_dependencies([st2]):
        pt = p.assign(s[0] * sm)
    return [st1, st2, pt]


def _velocity_step(p, grad, dt, indices=None):
    if isinstance(grad, tf.IndexedSlices):  # Sparse gradients.
        pt = p.scatter_sub(tf.IndexedSlices(grad * dt, indices))
    else:  # dense gradient
        pt = p.assign_sub(grad * dt)
    return [pt]


def _position_step(p, var, sm, dt):
    vart = var.assign_add(p / sm**2 * dt)
    return [vart]


class GLEOPT(Optimizer):
    """
    gradient-based EKF, this should be equivalent with EKF, but faster

    Args:
        learning_rate: learning rate
        epsilon: scale initial guess for P matrix
        q_0: initial process noise
        q_tau: time constant for noise
        q_min: minimal noise
        div_prec (str): dtype for division
    """

    def __init__(
        self,
        learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        A="OS-NS4-R4",
        C=None,
        mass="adam",
        tbath="adam",
        omega="adam",
        seed=1,
        v_verlet=True,
        name="GLEOPT",
        **kwargs,
    ):
        # input variables
        super().__init__(name=name, **kwargs)
        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        # Compute A or use preset, in any case, the actual A used is A*omega
        if not isinstance(A, str):
            self.A = np.array(A)
        else:
            self.A = np.array(_A_PRESETS[A])
        # Like A, the actual C used in T*tbath
        if C is None:
            self.C = np.eye(self.A.shape[0])
        else:
            self.C = np.array(C)
        # Set phys. quant.
        self.omega = omega
        self.tbath = tbath
        self.mass = mass
        # Setup adam flag
        self.adam = any((tbath == "adam", mass == "adam", omega == "adam"))
        # Misc configurations
        self.v_verlet = v_verlet
        self.seed = seed

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, "m")
            self.add_slot(var, "p")
            self.add_slot(
                var, "s", shape=tf.concat([self.A.shape[:1], tf.shape(var)], 0)
            )
            self.add_slot(var, "v")

    def _resource_apply_generic(self, grad, var, indices=None, apply_state=None):
        from tensorflow.keras.optimizers.schedules import deserialize

        floatx = tf.keras.backend.floatx()

        t = tf.cast(self.iterations + 1, floatx)
        try:
            alpha = deserialize(self.learning_rate)(t)
        except:
            alpha = tf.cast(self.learning_rate, floatx)
        m = self.get_slot(var, "m")
        p = self.get_slot(var, "p")
        s = self.get_slot(var, "s")
        v = self.get_slot(var, "v")
        xi = tf.random.normal(shape=tf.shape(s))

        if indices is None:
            vt = v.assign(v * self.beta_2 + (1 - self.beta_2) * grad**2)
        else:
            vt1 = v.assign(v * self.beta_2)
            with tf.control_dependencies([vt1]):
                vt2 = v.scatter_add(
                    tf.IndexedSlices((1 - self.beta_2) * grad**2, indices)
                )
            vt = tf.group([vt1, vt2])

        bias = 1.0 - tf.pow(self.beta_2, t)
        sv = tf.sqrt(v / bias) + 1e-7  # unbiased estimation of sqrt(v)

        if self.omega == "adam":
            omega = 1 - self.beta_1
        else:
            omega = float(self.omega)
        omega = tf.cast(omega, floatx)
        if self.tbath == "adam":
            tbath = 0.5 * sv * alpha
        elif float(self.tbath) < 0:
            tbath = -0.5 * float(self.tbath) * sv * alpha
        else:  # treat as constant temperature
            tbath = self.tbath
        tbath = tf.cast(tbath, floatx)
        if self.mass != "adam":
            sm = tf.sqrt(tf.cast(float(self.mass), floatx))
        else:
            sm = tf.sqrt(sv / (alpha * (1 - self.beta_1)))
        A = self.A * omega
        step = 0.5 if self.v_verlet else 1.0
        T = tf.linalg.expm(-step * A)
        C = tf.einsum("...,ij->...ij", tbath, self.C)
        SST = C - tf.einsum("ij,...jk,lk->...il", T, C, T)
        SST = tf.tensor_scatter_nd_sub(
            tf.einsum("...ij->ij...", SST), [[0, 0]], [(sv / sm) ** 2]
        )
        SST = tf.einsum("ij...->...ij", SST)
        S = tfp_cholesky(SST, tol=0.0)
        ops = [vt]
        with tf.control_dependencies(ops):
            ops += _thermostat_step(s, p, xi, T, S, sm)
        with tf.control_dependencies(ops):
            ops += _velocity_step(p, grad, 1.0, indices=indices)
        with tf.control_dependencies(ops):
            ops += _position_step(p, var, sm, 1.0)
        return tf.group(*ops)

    _resource_apply_sparse = _resource_apply_generic
    _resource_apply_dense = _resource_apply_generic

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self.learning_rate,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "A": self.A,
                "C": self.C,
                "omega": self.omega,
                "mass": self.mass,
                "tbath": self.tbath,
                "seed": self.seed,
            }
        )
        return config
