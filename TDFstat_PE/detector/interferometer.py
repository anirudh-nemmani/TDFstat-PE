import os
import struct

import numpy as np
from scipy.signal.windows import tukey

from ..utils import PropertyAccessor
from ..signal import TimeSeries

class Geometry:
    """Geometry of an interferometer."""

    def __init__(
        self,
        length,
        latitude,
        longitude,
        elevation,
        xarm_azimuth,
        yarm_azimuth,
        xarm_tilt,
        yarm_tilt,
        orientation,
    ):
        self.length = float(length)
        self.latitude = float(latitude)
        self.latitude_radians = np.radians(latitude)
        self.longitude = float(longitude)
        self.longitude_radians = np.radians(longitude)
        self.elevation = float(elevation)
        self.xarm_azimuth = float(xarm_azimuth)
        self.yarm_azimuth = float(yarm_azimuth)
        self.xarm_tilt = float(xarm_tilt)
        self.yarm_tilt = float(yarm_tilt)
        self.orientation = float(orientation)
        self.orientation_radians = np.radians(orientation)

class StrainData:
    """Time and frequency domain strain data."""

    def __init__(self, name, segment, band, delta_t, overlap):
        self.detector_name = name
        self.segment = segment
        self.band = band
        self.overlap = overlap

        self.delta_t = float(delta_t)
        self.start_time = 0.0
        self.duration = 0.0

        self.time_domain_strain = None
        self.frequency_domain_strain = None

        self.starting_frequency = (
            10.0 + (1.0 - overlap) * band / (2.0 * self.delta_t)
        )
        self.bandwidth = 1.0 / (2.0 * self.delta_t)
        self.sampling_frequency = None

        self.nlength = 0
        self.Nzeros = 0
        self.crf0 = 0.0
        self.mean = 0.0
        self.var = 0.0

        self.roll_off = None
        self.window_factor = None

    def load_time_domain_strain(self, strain_data_dir):
        """
        Load time-domain strain data from a binary file.

        Args:
            strain_data_dir (str): Directory containing the strain data files.
        """
        self.strain_data_dir = strain_data_dir
        filename = (
            f"{strain_data_dir}/{self.segment:03d}/"
            f"{self.detector_name}/xdat_"
            f"{self.segment:03d}_{self.band:04d}.bin"
        )

        try:
            with open(filename, "rb") as f:
                self.time_domain_strain = np.frombuffer(
                    f.read(), dtype=np.float32
                )
        except FileNotFoundError:
            raise FileNotFoundError(f"Strain file not found: {filename}")

        self._compute_statistics()
        self.duration = (self.nlength - 1) * self.delta_t
        self._read_start_time()
        self._generate_time_array()

    def generate_gaussian_strain_data(
        self, start_time, duration, variance, mean=0.0, seed=99
    ):
        self.start_time = float(start_time)
        self.duration = float(duration)
        self.nlength = int(duration / self.delta_t) + 1

        rng = np.random.default_rng(seed)
        self.time_domain_strain = rng.normal(
            loc=mean,
            scale=np.sqrt(variance),
            size=self.nlength,
        ).astype(np.float32)

        self._compute_statistics()
        self._generate_time_array()

    def generate_zero_strain_data(self, start_time, duration):
        self.start_time = float(start_time)
        self.duration = float(duration)
        self.nlength = int(duration / self.delta_t) + 1

        self.time_domain_strain = np.zeros(self.nlength, dtype=np.float32)
        self.Nzeros = self.nlength
        self.crf0 = np.inf
        self.mean = 0.0
        self.var = 0.0

        self._generate_time_array()

    @property
    def alpha(self):
        if self.roll_off is None:
            return 0.0
        return self.roll_off / (self.duration / 2.0)

    def time_domain_window(self, roll_off=None, alpha=None):
        """
        Apply a Tukey window to the time-domain strain data.

        Reference:
            https://dcc.ligo.org/DocDB/0027/T040089/000/T040089-00.pdf

        Args:
            roll_off (float, optional): Roll-off factor for the Tukey window.
            alpha (float, optional): Alpha parameter for the Tukey window.

        Returns:
            np.ndarray: Tukey window applied to the strain data.
        """
        if roll_off is not None:
            self.roll_off = roll_off
        elif alpha is not None:
            self.roll_off = alpha * self.duration

        window = tukey(self.nlength, alpha=self.alpha)
        self.window_factor = np.mean(window ** 2)
        return window

    def calculate_frequency_domain_strain(self, roll_off=0.4, alpha=None):
        if self.time_domain_strain is None:
            raise RuntimeError("Time-domain strain not available")

        if self.sampling_frequency is None:
            self.sampling_frequency = 1.0 / self.delta_t

        window = self.time_domain_window(roll_off, alpha)
        fft = np.fft.rfft(self.time_domain_strain * window)
        self.frequency_domain_strain = fft / self.sampling_frequency

        self.frequency_array = (
            self.starting_frequency
            + np.linspace(
                0.0,
                self.sampling_frequency / 2.0,
                len(self.frequency_domain_strain),
            )
        )

    def _generate_time_array(self):
        self.time_array = np.arange(0.0, self.duration, self.delta_t) + self.start_time
        self.time_square_array = self.time_array ** 2
        self.t_plus_t2_over_2 = self.time_array + 0.5 * self.time_square_array

    def _compute_statistics(self):
        self.nlength = len(self.time_domain_strain)
        self.Nzeros = np.count_nonzero(self.time_domain_strain == 0)

        if self.Nzeros < self.nlength:
            self.crf0 = self.nlength / (self.nlength - self.Nzeros)
        else:
            self.crf0 = np.inf

        self.mean = float(np.mean(self.time_domain_strain))
        self.var = float(
            self.crf0 * np.mean((self.time_domain_strain - self.mean) ** 2)
        )

    def _read_start_time(self):
        filename = (
            f"{self.strain_data_dir}/{self.segment:03d}/"
            f"{self.detector_name}/starting_date"
        )
        with open(filename, "r") as f:
            self.start_time = float(f.read().strip())

class Ephemeris:
    """Detector ephemeris."""

    def __init__(self, name, segment, band, delta_t):
        self.detector_name = name
        self.segment = segment
        self.band = band
        self.delta_t = float(delta_t)

        self.start_time = 0.0
        self.duration = 0.0
        self.nlength = 0

        self.DetSSB = None
        self.phir = 0.0
        self.epsm = 0.0

    def load_detector_ephemeris(self, ephemeris_dir):
        self.ephemeris_dir = ephemeris_dir
        filename = (
            f"{ephemeris_dir}/{self.segment:03d}/"
            f"{self.detector_name}/DetSSB.bin"
        )

        with open(filename, "rb") as f:
            buffer = f.read()

        self.DetSSB = np.frombuffer(
            buffer[:-16], dtype=np.float64
        ).reshape(-1, 3)
        self.phir, self.epsm = struct.unpack("dd", buffer[-16:])

        self.nlength = self.DetSSB.shape[0]
        self.duration = (self.nlength - 1) * self.delta_t
        self._read_start_time()

    def _read_start_time(self):
        filename = (
            f"{self.ephemeris_dir}/{self.segment:03d}/"
            f"{self.detector_name}/starting_date"
        )
        with open(filename, "r") as f:
            self.start_time = float(f.read().strip())

class Interferometer(object):
    """
    Class to manage interferometer Geometry, Strain data and Ephemeris.
    """

    length = PropertyAccessor("geometry", "length")
    latitude = PropertyAccessor("geometry", "latitude")
    latitude_radians = PropertyAccessor("geometry", "latitude_radians")
    longitude = PropertyAccessor("geometry", "longitude")
    longitude_radians = PropertyAccessor("geometry", "longitude_radians")
    elevation = PropertyAccessor("geometry", "elevation")
    xarm_azimuth = PropertyAccessor("geometry", "xarm_azimuth")
    yarm_azimuth = PropertyAccessor("geometry", "yarm_azimuth")
    xarm_tilt = PropertyAccessor("geometry", "xarm_tilt")
    yarm_tilt = PropertyAccessor("geometry", "yarm_tilt")
    orientation = PropertyAccessor("geometry", "orientation")
    orientation_radians = PropertyAccessor("geometry", "orientation_radians")

    detector_name = PropertyAccessor("strain_data", "detector_name")
    strain_data_dir = PropertyAccessor("strain_data", "strain_data_dir")
    ephemeris_dir = PropertyAccessor("ephemeris", "ephemeris_dir")

    start_time = PropertyAccessor("ephemeris", "start_time")
    delta_t = PropertyAccessor("ephemeris", "delta_t")
    duration = PropertyAccessor("ephemeris", "duration")

    starting_frequency = PropertyAccessor("strain_data", "starting_frequency")
    sampling_frequency = PropertyAccessor("strain_data", "sampling_frequency")
    bandwidth = PropertyAccessor("strain_data", "bandwidth")

    time_domain_strain = PropertyAccessor("strain_data", "time_domain_strain")
    frequency_domain_strain = PropertyAccessor("strain_data", "frequency_domain_strain")

    time_array = PropertyAccessor("strain_data", "time_array")
    time_square_array = PropertyAccessor("strain_data", "time_square_array")
    t_plus_t2_over_2 = PropertyAccessor("strain_data", "t_plus_t2_over_2")
    frequency_array = PropertyAccessor("strain_data", "frequency_array")

    segment = PropertyAccessor("ephemeris", "segment")
    band = PropertyAccessor("ephemeris", "band")
    overlap = PropertyAccessor("strain_data", "overlap")

    DetSSB = PropertyAccessor("ephemeris", "DetSSB")
    nlength = PropertyAccessor("ephemeris", "nlength")
    phir = PropertyAccessor("ephemeris", "phir")
    epsm = PropertyAccessor("ephemeris", "epsm")

    def __init__(
        self,
        name,
        length,
        latitude,
        longitude,
        elevation,
        xarm_azimuth,
        yarm_azimuth,
        xarm_tilt,
        yarm_tilt,
        orientation,
        segment,
        band,
        delta_t,
        overlap,
    ):
        self.name = name
        self.geometry = Geometry(
            length,
            latitude,
            longitude,
            elevation,
            xarm_azimuth,
            yarm_azimuth,
            xarm_tilt,
            yarm_tilt,
            orientation,
        )
        self.strain_data = StrainData(self.name, segment, band, delta_t, overlap)

        self.ephemeris = Ephemeris(self.name, segment, band, delta_t)

    def load_detector_ephemeris(self, ephemeris_dir):
        self.ephemeris.load_detector_ephemeris(ephemeris_dir)

    def load_time_domain_strain(self, strain_data_dir):
        self.strain_data.load_time_domain_strain(strain_data_dir)

    def generate_gaussian_strain_data(self, start_time, duration, variance, mean=0.0, seed=99):
        self.strain_data.generate_gaussian_strain_data(start_time, duration, variance, mean, seed)

    def generate_zero_strain_data(self, start_time, duration):
        self.strain_data.generate_zero_strain_data(start_time, duration)

    def calculate_frequency_domain_strain(self):
        self.strain_data.calculate_frequency_domain_strain()

    def inject_signal(self, **params):
        self.template = TimeSeries(self)
        self.template.generate_time_domain_signal(**params)


def load_interferometer(filename, segment, band, delta_t, overlap):
    """
    Load interferometer properties from a configuration file.
    """
    parameters = {}
    with open(filename, "r") as parameter_file:
        for line in parameter_file:
            if line.startswith("#") or line.strip() == "":
                continue
            key, value = map(str.strip, line.split("=", 1))
            parameters[key] = eval(value)

    parameters.update(
        {
            "segment": segment,
            "band": band,
            "delta_t": delta_t,
            "overlap": overlap,
        }
    )
    return Interferometer(**parameters)


def get_empty_interferometer(name, segment, band, delta_t, overlap):
    """
    Create an empty interferometer object for a given detector.
    """
    filename = os.path.join(
        os.path.dirname(__file__), "detectors", "{}.interferometer".format(name)
    )
    try:
        return load_interferometer(filename, segment, band, delta_t, overlap)
    except OSError:
        raise ValueError(f"Interferometer {name} not implemented")


class InterferometerList(list):
    """
    A list of Interferometer objects with additional utility methods.
    """

    def __init__(self, interferometers, segment, band, delta_t, overlap):
        """
        Initialize the InterferometerList with a list of Interferometer objects
        """
        super().__init__()
        if not isinstance(interferometers, (list, tuple)):
            raise TypeError("Input must be a list or tuple of interferometers")
        for ifo in interferometers:
            if isinstance(ifo, str):
                ifo = get_empty_interferometer(ifo, segment, band, delta_t, overlap)
            if not isinstance(ifo, Interferometer):
                raise TypeError(
                    "All items in the input list must be Interferometer objects or valid names"
                )
            else:
                self.append(ifo)

    def load_detector_ephemeris(self, ephemeris_dir):
        for ifo in self:
            ifo.load_detector_ephemeris(ephemeris_dir)

    def load_time_domain_strain(self, strain_data_dir):
        for ifo in self:
            ifo.load_time_domain_strain(strain_data_dir)

    def calculate_frequency_domain_strain(self):
        for ifo in self:
            ifo.calculate_frequency_domain_strain()

    def generate_gaussian_strain_data(self, start_time, duration, variance, mean=0.0, seed=99):
        for ifo in self:
            ifo.generate_gaussian_strain_data(start_time, duration, variance, mean, seed)

    def generate_zero_strain_data(self, start_time, duration):
        for ifo in self:
            ifo.generate_zero_strain_data(start_time, duration)

    def inject_signal(self, **params):
        for ifo in self:
            ifo.inject_signal(**params)
