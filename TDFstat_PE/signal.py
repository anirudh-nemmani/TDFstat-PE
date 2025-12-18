import constants
import numpy as np

class TimeSeries:
    def __init__(self, detector):
        self.detector = detector

        self.segment = detector.segment
        self.band = detector.band
        self.overlap = detector.overlap

        self.start_time = detector.start_time
        self.delta_t = detector.delta_t
        self.duration = detector.duration

        self.starting_frequency = detector.starting_frequency
        self.sampling_frequency = detector.sampling_frequency
        self.bandwidth = detector.bandwidth

        self.time_array = detector.time_array
        self.time_square_array = detector.time_square_array
        self.t_plus_t_squ_by_2 = detector.t_plus_t2_over_2

    @staticmethod
    def _generate_source_vector(ra, dec):
        """
        Generate a constant unit vector in the direction of the star in the SSB reference frame
        """
        cos_dec = np.cos(dec)
        return np.array([
            np.cos(ra) * cos_dec,
            np.sin(ra) * cos_dec,
            np.sin(dec),
        ])

    def _generate_amplitude_modulation(self, ra, dec):
        """
        Generate the amplitude modulation functions a(t) and b(t).

        Reference:
            Phys. Rev. D 58, 063001 (1998), Eq. 12–13
        """
        ra_mod = (
            ra
            - self.detector.ephemeris.phir
            - constants.C_OMEGA_R * self.time_array
        )

        sin_2psi = np.sin(2.0 * self.detector.orientation_radians)
        cos_2psi = np.cos(2.0 * self.detector.orientation_radians)

        lon = self.detector.longitude_radians
        sin_lon = np.sin(lon)
        cos_lon = np.cos(lon)
        sin_2lon = np.sin(2.0 * lon)
        cos_2lon = np.cos(2.0 * lon)

        sin_dec = np.sin(dec)
        cos_dec = np.cos(dec)
        sin_2dec = np.sin(2.0 * dec)
        cos_2dec = np.cos(2.0 * dec)

        cos_ra = np.cos(ra_mod)
        sin_ra = np.sin(ra_mod)
        cos_2ra = np.cos(2.0 * ra_mod)
        sin_2ra = np.sin(2.0 * ra_mod)

        a = (
            0.0625 * sin_2psi * (3.0 - cos_2lon)
            * (3.0 - cos_2dec) * cos_2ra
            - 0.25 * cos_2psi * sin_lon
            * (3.0 - cos_2dec) * sin_2ra
            + 0.25 * sin_2psi * sin_2lon
            * sin_2dec * cos_ra
            - 0.5 * cos_2psi * cos_lon
            * sin_2dec * sin_ra
            + 0.75 * sin_2psi * cos_lon ** 2
            * cos_dec ** 2
        )

        b = (
            cos_2psi * sin_lon * sin_dec * cos_2ra
            + 0.25 * sin_2psi * (3.0 - cos_2lon)
            * sin_dec * sin_2ra
            + cos_2psi * cos_lon * cos_dec * cos_ra
            + 0.25 * sin_2psi * sin_2lon
            * cos_dec * sin_ra
        )
        return a, b

    @staticmethod
    def _generate_four_amplitudes(
        wobble_angle, inclination, polarization, reference_phase
    ):
        """
        Generate the four amplitudes of the continuous gravitational
        wave signal.

        Reference:
            Phys. Rev. D 58, 063001 (1998), Eq. 32–35
        """
        sin_w2 = np.sin(wobble_angle) ** 2
        cos_i = np.cos(inclination)
        inc_term = 0.5 * (1.0 + cos_i ** 2)

        cos_2psi = np.cos(2.0 * polarization)
        sin_2psi = np.sin(2.0 * polarization)
        cos_2phi = np.cos(2.0 * reference_phase)
        sin_2phi = np.sin(2.0 * reference_phase)

        a21 = sin_w2 * (
            inc_term * cos_2psi * cos_2phi
            - cos_i * sin_2psi * sin_2phi
        )
        a22 = sin_w2 * (
            inc_term * sin_2psi * cos_2phi
            + cos_i * cos_2psi * sin_2phi
        )
        a23 = -sin_w2 * (
            inc_term * cos_2psi * sin_2phi
            + cos_i * sin_2psi * cos_2phi
        )
        a24 = -sin_w2 * (
            inc_term * sin_2psi * sin_2phi
            - cos_i * cos_2psi * cos_2phi
        )

        return a21, a22, a23, a24

    def _generate_signal_phase(self, frequency, spin_down, ra, dec):
        """
        Generate the phase of the continuous gravitational wave signal.
        Reference:
            http://dx.doi.org/10.1103/PhysRevD.58.063001 eq 14
        """
        nsource = self._generate_source_vector(ra, dec)
        r_dot_n = np.dot(self.detector.ephemeris.DetSSB, nsource)

        zero_order = (1 + r_dot_n) * frequency * self.time_array
        first_order = self.t_plus_t_squ_by_2 * spin_down
        return 2 * np.pi * (zero_order + first_order)

    def generate_time_domain_signal(self, **params):
        """
        Generate the time-domain continuous gravitational wave signal.
        """
        phase = self._generate_signal_phase(
            params["frequency"],
            params["spin_down"],
            params["ra"],
            params["dec"],
        )

        a, b = self._generate_amplitude_modulation(
            params["ra"], params["dec"]
        )

        a21, a22, a23, a24 = self._generate_four_amplitudes(
            params["wobble_angle"],
            params["inclination"],
            params["polarization"],
            params["reference_phase"],
        )

        signal = (
            (a * a21 + b * a22) * np.cos(2.0 * phase)
            + (a * a23 + b * a24) * np.sin(2.0 * phase)
        )

        return signal, self.time_array
