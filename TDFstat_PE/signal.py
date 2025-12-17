import configparser

import numpy as np

import constants
from detector import det_init


class Signal:
    def __init__(self, params):
        for i in params["detectors"]:
            self.detector = det_init.Detector(
                i, params["indir"], params["seg"], params["band"], params["N"]
            )

            self._generate_time_array()
            self._generate_signal()

    def _generate_time_array(self):
        dt = self.sampling_time()
        start_time = self.detector.start_time
        self.time_array = (
            start_time
            + np.arange(0, self.duration + dt, dt)
            - self.params["reference_time"]
        )
        self.time_square_array = self.time_array * self.time_array
        self.t_plus_t_squ_by_2 = self.time_array + self.time_square_array / 2

    def nSource(self):
        n1 = np.cos(self.params["ra"]) * np.cos(self.params["dec"])
        n2 = np.sin(self.params["ra"]) * np.cos(self.params["dec"])
        n3 = np.sin(self.params["dec"])

        return np.array([n1, n2, n3])

    # Amplitude modulation function ( a replica of modvir in settings.c)

    def amplitude_modulation(self):
        modulation = self.ra - self.phir - self.omega_r * t
        a1 = (
            (1 / 16)
            * np.sin(2 * self.detector.egam)
            * (3 - np.cos(2 * self.detector.elam))
            * (3 - np.cos(2 * self.dec))
            * np.cos(2 * (modulation))
        )
        a2 = (
            -0.25
            * np.cos(2 * self.detector.egam)
            * np.sin(self.detector.elam)
            * (3 - np.cos(2 * self.dec))
            * np.sin(2 * (modulation))
        )
        a3 = (
            0.25
            * np.sin(2 * self.detector.egam)
            * np.sin(2 * self.detector.elam)
            * np.sin(2 * self.dec)
            * np.cos(modulation)
        )
        a4 = (
            -0.5
            * np.cos(2 * self.detector.egam)
            * np.cos(self.detector.elam)
            * np.sin(2 * self.dec)
            * np.sin(modulation)
        )
        a5 = (
            0.75
            * np.sin(2 * self.detector.egam)
            * (np.cos(self.detector.elam)) ** 2
            * (np.cos(self.dec)) ** 2
        )

        a = a1 + a2 + a3 + a4 + a5

        b1 = (
            np.cos(2 * self.detector.egam)
            * np.sin(self.detector.elam)
            * np.sin(self.dec)
            * np.cos(2 * modulation)
        )
        b2 = (
            0.25
            * np.sin(2 * self.detector.egam)
            * (3 - np.cos(2 * self.detector.elam))
            * np.sin(self.dec)
            * np.sin(2 * modulation)
        )
        b3 = (
            np.cos(2 * self.detector.egam)
            * np.cos(self.detector.elam)
            * np.cos(self.dec)
            * np.cos(modulation)
        )
        b4 = (
            0.25
            * np.sin(2 * self.detector.egam)
            * np.sin(2 * self.detector.elam)
            * np.cos(self.self.dec)
            * np.sin(modulation)
        )

        b = b1 + b2 + b3 + b4

        return a, b

    def OMEGA_S(self):
        """
        Dimensionless angular frequency
        """

        return 2 * np.pi * self.frequency * self.sampling_time

    def four_amplitudes(self):
        # To be cross-verified with TDFstat paper 1 (eq.32,33,34,35)

        term1 = np.sin(self.wobble_angle) * np.sin(self.wobble_angle)
        term2 = 0.5 * (1 + np.cos(self.inclination) * np.cos(self.inclination))

        psi = self.polarization
        phi0 = self.phase
        iota = self.inclination

        A21 = term1 * (
            term2 * np.cos(2 * psi) * np.cos(2 * phi0)
            - np.cos(iota) * np.sin(2 * psi) * np.sin(2 * phi0)
        )
        A22 = term1 * (
            term2 * np.sin(2 * psi) * np.cos(2 * phi0)
            + np.cos(iota) * np.cos(2 * psi) * np.sin(2 * phi0)
        )

        A23 = term1 * (
            -term2 * np.cos(2 * psi) * np.sin(2 * phi0)
            - np.cos(iota) * np.sin(2 * psi) * np.cos(2 * phi0)
        )
        A24 = term1 * (
            -term2 * np.sin(2 * psi) * np.sin(2 * phi0)
            + np.cos(iota) * np.cos(2 * psi) * np.cos(2 * phi0)
        )

        return A21, A22, A23, A24

    def phase_function(self):
        """Contains the phase part from Eq (14) of PhysRevD.58.063001 . This function should be a function of time."""

        nsource = self.nSource()

        r_dot_n = self.detector.DetSSB.dot(nsource)

        zero_order = (1 + r_dot_n) * self.params["frequency"] * self.time_array

        first_order = self.t_plus_t_squ_by_2 * self.params["spin_down"]

        phase = self.params["phase"] + 2 * np.pi * (zero_order + first_order)

        return phase

    def signal(self):
        A21, A22, A23, A24 = self.four_amplitudes()
        a, b = self.amplitude_modulation()

        return (a * A21 + b * A22) * np.cos(self.phase_function()) + (
            a * A23 + b * A24
        ) * np.sin(self.phase_function())
