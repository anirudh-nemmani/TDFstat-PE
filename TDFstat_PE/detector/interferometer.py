import os
import struct

import numpy as np

from ..utils import PropertyAccessor


class Geometry(object):
    """
    Class to manage the geometric properties of an interferometer.

    Attributes:
        length (float): Length of the interferometer's arms.
        latitude (float): Latitude of the interferometer's location.
        longitude (float): Longitude of the interferometer's location.
        elevation (float): Elevation of the interferometer's location.
        xarm_azimuth (float): Azimuth angle of the x arm.
        yarm_azimuth (float): Azimuth angle of the y arm.
        xarm_tilt (float): Tilt angle of the x arm.
        yarm_tilt (float): Tilt angle of the y arm.
        orientation (float): Orientation angle of the interferometer.
    """

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
        self.length = length
        self.latitude = latitude
        self.latitude_radians = np.radians(latitude)
        self.longitude = longitude
        self.longitude_radians = np.radians(longitude)
        self.elevation = elevation
        self.xarm_azimuth = xarm_azimuth
        self.yarm_azimuth = yarm_azimuth
        self.xarm_tilt = xarm_tilt
        self.yarm_tilt = yarm_tilt
        self.orientation = orientation
        self.orientation_radians = np.radians(orientation)


class StrainData(object):
    """
    Class to manage the strain data of an interferometer.

    Attributes:
        detector_name (str): Name of the interferometer.
        time_domain_strain (np.ndarray): Strain data in the time domain.
        frequency_domain_strain (np.ndarray): Strain data in the frequency domain.
        sampling_frequency (float): Sampling frequency of the strain data.
        duration (float): Duration of the strain data segment.
        start_time (float): Start time of the strain data segment.
        segment (int): Segment number of the data.
        band (int): Frequency band number of the data.
        nlength (int): Number of data points in the strain data.
        DetSSB (np.ndarray): Ephemeris data of the detector.
        phir (float): Phase parameter of the detector.
        epsm (float): Eccentricity parameter of the detector.
        Nzeros (int): Number of zero values in the strain data.
        crf0 (float): Correction factor for zero values.
        mean (float): Mean value of the strain data.
        var (float): Variance of the strain data.
    """

    def __init__(self, name, segment, band, delta_t, overlap):
        self.detector_name = name
        self.segment = segment
        self.band = band
        self.overlap = overlap

        self.start_time = 0.0
        self.delta_t = delta_t
        self.duration = 0.0
        self.time_domain_strain = None

        self.starting_frequency = 10 + (1 - self.overlap) * self.band / (
            2 * self.delta_t
        )
        self.sampling_frequency = None
        self.bandwidth = 1 / (2 * self.delta_t)
        self.frequency_domain_strain = None

        self.nlength = 0
        self.Nzeros = 0
        self.crf0 = 0.0
        self.mean = 0.0
        self.var = 0.0

    def load_time_domain_strain(self, strain_data_dir):
        """
        Load the time-domain strain data from a binary file.

        Args:
            strain_data_dir (str): Directory containing the strain data files.
        """
        self.strain_data_dir = strain_data_dir
        filename = f"{self.strain_data_dir}/{self.segment:03d}/{self.detector_name}/xdat_{self.segment:03d}_{self.band:04d}.bin"
        try:
            with open(filename, "rb") as data:
                self.time_domain_strain = np.frombuffer(data.read(), dtype=np.float32)

            self.nlength = len(self.time_domain_strain)
            self.Nzeros = np.count_nonzero(self.time_domain_strain == 0)
            if self.Nzeros < self.nlength:
                self.crf0 = self.nlength / (self.nlength - self.Nzeros)
            else:
                print("Warning: All data points are zero.")
                self.crf0 = float("inf")
            self.mean = np.mean(self.time_domain_strain)
            self.var = self.crf0 * np.mean((self.time_domain_strain - self.mean) ** 2)

            self.duration = (self.nlength - 1) * self.delta_t
            self.read_start_time()

        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            print(
                "Use the function generate_gaussian_strain_data or have a strain data file in the ephemeris directory."
            )
            return

    def generate_gaussian_strain_data(
        self, start_time, duration, variance, mean=0.0, seed=99
    ):
        """
        Generate synthetic Gaussian strain data.

        Args:
            start_time (float): Start time of the strain data segment.
            duration (float): Duration of the strain data segment in seconds.
            variance (float): Variance of the Gaussian noise.
            mean (float, optional): Mean of the Gaussian noise. Defaults to 0.0.
            seed (int, optional): Seed for the random number generator. Defaults to 99.
        """
        self.start_time = start_time
        self.duration = duration
        self.nlength = int(self.duration / self.delta_t) + 1

        rng = np.random.default_rng(seed)
        self.time_domain_strain = rng.normal(
            loc=mean, scale=np.sqrt(variance), size=self.nlength
        )

        self.Nzeros = np.count_nonzero(self.time_domain_strain == 0)
        if self.Nzeros < self.nlength:
            self.crf0 = self.nlength / (self.nlength - self.Nzeros)
        else:
            print("Warning: All data points are zero.")
            self.crf0 = float("inf")
        self.mean = np.mean(self.time_domain_strain)
        self.var = self.crf0 * np.mean((self.time_domain_strain - self.mean) ** 2)

    @property
    def alpha(self):
        """
        Calculate the alpha parameter for the Tukey window.

        Returns:
            float: Alpha parameter for the Tukey window.
        """
        return self.roll_off / (self.duration / 2)

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
        from scipy.signal.windows import tukey

        if roll_off is not None:
            self.roll_off = roll_off
        elif alpha is not None:
            self.roll_off = 2 * alpha * self.duration / 2

        window = tukey(len(self.time_domain_strain), alpha=self.alpha)
        self.window_factor = np.mean(window**2)
        return window

    def calculate_frequency_domain_strain(self, roll_off=0.4, alpha=None):
        """
        Compute the frequency-domain strain data using FFT.

        Args:
            roll_off (float, optional): Roll-off factor for the Tukey window.
            alpha (float, optional): Alpha parameter for the Tukey window.
        """
        if self.sampling_frequency is None:
            self.sampling_frequency = 1.0 / self.delta_t
        window = self.time_domain_window(roll_off=roll_off, alpha=alpha)
        if self.time_domain_strain is None:
            print("Error: Time-domain strain data not loaded.")
            return

        self.frequency_domain_strain = np.fft.rfft(self.time_domain_strain * window)
        self.frequency_domain_strain /= self.sampling_frequency

        self.frequency_array = self.starting_frequency + np.linspace(
            0, self.sampling_frequency / 2, len(self.frequency_domain_strain)
        )

    def read_start_time(self):
        """
        Read the start time of the strain data segment from a text file.

        Args:
            name (str): Name of the interferometer.
        """
        filename = f"{self.strain_data_dir}/{self.segment:03d}/{self.detector_name}/starting_date"
        try:
            with open(filename, "r") as data:
                self.start_time = float(data.read().strip())
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return


class Ephemeris(object):
    """
    Class to manage the ephemeris data of a detector.

    Attributes:
        DetSSB (np.ndarray): Ephemeris data of the detector.
        phir (float): Phase parameter of the detector.
        epsm (float): Eccentricity parameter of the detector.
    """

    def __init__(self, name, segment, band, delta_t):
        self.detector_name = name
        self.segment = segment
        self.band = band
        self.delta_t = delta_t
        self.nlength = 0
        self.duration = 0.0
        self.start_time = 0.0

        self.DetSSB = None
        self.phir = 0.0
        self.epsm = 0.0

    def load_detector_ephemeris(self, ephemeris_dir):
        """
        Load the detector ephemeris data from a binary file.

        Args:
            name (str): Name of the interferometer.
        """
        self.ephemeris_dir = ephemeris_dir
        filename = (
            f"{self.ephemeris_dir}/{self.segment:03d}/{self.detector_name}/DetSSB.bin"
        )
        try:
            with open(filename, "rb") as data:
                buffer = data.read()
                # Reading all but the last two floats for DetSSB
                self.DetSSB = np.frombuffer(buffer[:-16], dtype=np.float64).reshape(
                    -1, 3
                )
                # Reading the last two floats for phir and epsm
                self.phir, self.epsm = struct.unpack("dd", buffer[-16:])

            self.nlength = self.DetSSB.shape[0]
            self.duration = (self.DetSSB.shape[0] - 1) * self.delta_t
            self.read_start_time()
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return

    def read_start_time(self):
        """
        Read the start time of the ephemeris data segment from a text file.

        Args:
            name (str): Name of the interferometer.
        """
        filename = f"{self.ephemeris_dir}/{self.segment:03d}/{self.detector_name}/starting_date"
        try:
            with open(filename, "r") as data:
                self.start_time = float(data.read().strip())
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return


class Interferometer(object):
    """
    Class to manage interferometer Geometry, Strain data and Ephemeris.
    """

    length = PropertyAccessor("geometry", "length")
    latitude = PropertyAccessor("geometry", "latitude")
    latitude_radians = PropertyAccessor(
        "geometry", "latitude_radians"
    )  # Ephi in TDFstat
    longitude = PropertyAccessor("geometry", "longitude")
    longitude_radians = PropertyAccessor(
        "geometry", "longitude_radians"
    )  # ELambda in TDFstat
    elevation = PropertyAccessor("geometry", "elevation")  # Height in TDFstat
    xarm_azimuth = PropertyAccessor("geometry", "xarm_azimuth")
    yarm_azimuth = PropertyAccessor("geometry", "yarm_azimuth")
    xarm_tilt = PropertyAccessor("geometry", "xarm_tilt")
    yarm_tilt = PropertyAccessor("geometry", "yarm_tilt")
    orientation = PropertyAccessor("geometry", "orientation")  # Gamma in TDFstat
    orientation_radians = PropertyAccessor("geometry", "orientation_radians")

    detector_name = PropertyAccessor("strain_data", "detector_name")
    strain_data_dir = PropertyAccessor("strain_data", "strain_data_dir")

    start_time = PropertyAccessor("strain_data", "start_time")
    delta_t = PropertyAccessor("strain_data", "delta_t")
    duration = PropertyAccessor("strain_data", "duration")
    time_domain_strain = PropertyAccessor("strain_data", "time_domain_strain")

    starting_frequency = PropertyAccessor("strain_data", "starting_frequency")
    sampling_frequency = PropertyAccessor("strain_data", "sampling_frequency")
    bandwidth = PropertyAccessor("strain_data", "bandwidth")
    frequency_domain_strain = PropertyAccessor("strain_data", "frequency_domain_strain")

    segment = PropertyAccessor("strain_data", "segment")
    band = PropertyAccessor("strain_data", "band")
    overlap = PropertyAccessor("strain_data", "overlap")

    nlength = PropertyAccessor("strain_data", "nlength")
    Nzeros = PropertyAccessor("strain_data", "Nzeros")
    crf0 = PropertyAccessor("strain_data", "crf0")
    mean = PropertyAccessor("strain_data", "mean")
    var = PropertyAccessor("strain_data", "var")

    detector_name = PropertyAccessor("ephemeris", "detector_name")
    start_time = PropertyAccessor("ephemeris", "start_time")
    duration = PropertyAccessor("ephemeris", "duration")
    delta_t = PropertyAccessor("ephemeris", "delta_t")
    segment = PropertyAccessor("ephemeris", "segment")
    band = PropertyAccessor("ephemeris", "band")
    ephemeris_dir = PropertyAccessor("ephemeris", "ephemeris_dir")
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
        """
        Load the ephemeris data for the detector.
        """
        self.ephemeris.load_detector_ephemeris(ephemeris_dir)

    def load_time_domain_strain(self, strain_data_dir):
        """
        Load the time-domain strain data for the interferometer.
        """
        self.strain_data.load_time_domain_strain(strain_data_dir)

    def generate_gaussian_strain_data(
        self, start_time, duration, variance, mean=0.0, seed=99
    ):
        """
        Generate synthetic Gaussian strain data for the interferometer.
        """
        self.strain_data.generate_gaussian_strain_data(
            start_time, duration, variance, mean, seed
        )

    def calculate_frequency_domain_strain(self):
        """
        Compute the frequency-domain strain data for the interferometer.
        """
        self.strain_data.calculate_frequency_domain_strain()


def load_interferometer(filename, segment, band, delta_t, overlap):
    """
    Load interferometer properties from a configuration file.

    Args:
        filename (str): Path to the configuration file.
        segment (int): Segment number of the data.
        band (int): Frequency band number of the data.
        delta_t (float): Time step of the data.
        overlap (float): Overlap factor for the data.

    Returns:
        Interferometer: Loaded interferometer object.
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

    Args:
        name (str): Name of the interferometer.
        segment (int): Segment number of the data.
        band (int): Frequency band number of the data.
        delta_t (float): Time step of the data.
        overlap (float): Overlap factor for the data.

    Returns:
        Interferometer: Empty interferometer object.
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

    Args:
        interferometers (list): List of interferometer names or objects.
        segment (int): Segment number of the data.
        band (int): Frequency band number of the data.
        delta_t (float): Time step of the data.
        overlap (float): Overlap factor for the data.
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
        """
        Load the ephemeris data for all interferometers in the list.
        """
        for ifo in self:
            ifo.load_detector_ephemeris(ephemeris_dir)

    def load_time_domain_strain(self, strain_data_dir):
        """
        Load the time-domain strain data for all interferometers in the list.
        """
        for ifo in self:
            ifo.load_time_domain_strain(strain_data_dir)

    def calculate_frequency_domain_strain(self):
        """
        Compute the frequency-domain strain data for all interferometers in the list.
        """
        for ifo in self:
            ifo.calculate_frequency_domain_strain()

    def generate_gaussian_strain_data(
        self, start_time, duration, variance, mean=0.0, seed=99
    ):
        """
        Generate synthetic Gaussian strain data for all interferometers in the list.
        """
        for ifo in self:
            ifo.generate_gaussian_strain_data(
                start_time, duration, variance, mean, seed
            )
