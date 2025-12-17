import os
import struct

import numpy as np

from ..utils import PropertyAccessor


class Geometry(object):
    """
    Class to handle interferometer geometry properties.

    Attributes:
        length (float): Arm length of the interferometer.
        latitude (float): Latitude of the interferometer.
        longitude (float): Longitude of the interferometer.
        elevation (float): Elevation of the interferometer.
        xarm_azimuth (float): Azimuth of the x arm.
        yarm_azimuth (float): Azimuth of the y arm.
        xarm_tilt (float): Tilt of the x arm.
        yarm_tilt (float): Tilt of the y arm.
        orientation (float): Orientation of the interferometer.
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
    Class to handle interferometer strain data properties.

    Attributes:
        time_domain_strain (np.ndarray): Time-domain strain data.
        frequency_domain_strain (np.ndarray): Frequency-domain strain data.
        sampling_frequency (float): Sampling frequency of the strain data.
        duration (float): Duration of the strain data segment.
        start_time (float): Start time of the strain data segment.
        segment (int): Segment number.
        band (int): Frequency band number.
        nlength (int): Number of data points in the strain data.
        DetSSB (np.ndarray): Ephemeris of the detector.
        phir (float): Phase parameter.
        epsm (float): Eccentricity parameter.
        Nzeros (int): Number of zero values in the strain data.
        crf0 (float): Correction factor for zero values.
        mean (float): Mean of the strain data.
        var (float): Variance of the strain data.
    """

    def __init__(self, name, delta_t, segment, band, strain_data_dir):
        self.time_domain_strain = None
        self.frequency_domain_strain = None

        self.delta_t = delta_t
        self.sampling_frequency = 1.0 / delta_t
        self.duration = 0.0
        self.start_time = 0.0

        self.segment = segment
        self.band = band
        self.strain_data_dir = strain_data_dir

        self.nlength = 0
        self.Nzeros = 0
        self.crf0 = 0.0
        self.mean = 0.0
        self.var = 0.0

    def load_time_domain_strain(self, name):
        """
        Load the time-domain strain data from a binary file.

        Args:
            name (str): Name of the interferometer.
            segment (int): Segment number.
            band (int): Frequency band number.
        """
        filename = f"{self.strain_data_dir}/{self.segment:03d}/{name}/xdat_{self.segment:03d}_{self.band:04d}.bin"
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

            self.duration = self.nlength * self.delta_t

        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return

    def calculate_frequency_domain_strain(self):
        """
        Calculate the frequency-domain strain data using FFT.
        """
        if self.time_domain_strain is None:
            print("Error: Time-domain strain data not loaded.")
            return

        self.frequency_domain_strain = np.fft.rfft(self.time_domain_strain)

    def read_start_time(self, name):
        """
        Read the start time of the strain data segment from a text file.

        Args:
            name (str): Name of the interferometer.
            segment (int): Segment number.
        """
        filename = f"{self.strain_data_dir}/{self.segment:03d}/{name}/starting_date"
        try:
            with open(filename, "r") as data:
                self.start_time = float(data.read().strip())
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return


class Ephimeries(object):
    """
    Class to handle detector ephemeris data.

    Attributes:
        DetSSB (np.ndarray): Ephemeris of the detector.
        phir (float): Phase parameter.
        epsm (float): Eccentricity parameter.
    """

    def __init__(self, segment, band, delta_t, ephemeries_dir):
        self.segment = segment
        self.band = band
        self.delta_t = delta_t
        self.nlength = 0
        self.duration = 0.0
        self.start_time = 0.0
        self.ephemeries_dir = ephemeries_dir

        self.DetSSB = None
        self.phir = 0.0
        self.epsm = 0.0

    def load_detector_ephemeris(self, name):
        """
        Load the detector ephemeris data from a binary file.

        Args:
            name (str): Name of the interferometer.
            segment (int): Segment number.
        """
        filename = f"{self.ephemeries_dir}/{self.segment:03d}/{name}/DetSSB.bin"
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
            self.duration = self.DetSSB.shape[0] * self.delta_t
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return

    def read_start_time(self, name):
        """
        Read the start time of the strain data segment from a text file.

        Args:
            name (str): Name of the interferometer.
            segment (int): Segment number.
        """
        filename = f"{self.ephemeries_dir}/{self.segment:03d}/{name}/starting_date"
        try:
            with open(filename, "r") as data:
                self.start_time = float(data.read().strip())
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return


class Interferometer(object):
    """
    Class to handle interferometer data and properties.

    Attributes:
        name (str): Name of the interferometer.
        latitude (float): Latitude of the interferometer.
        longitude (float): Longitude of the interferometer.
        elevation (float): Elevation of the interferometer.
        arm_length (float): Arm length of the interferometer.
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

    strain_data_dir = PropertyAccessor("strain_data", "strain_data_dir")
    time_domain_strain = PropertyAccessor("strain_data", "time_domain_strain")
    frequency_domain_strain = PropertyAccessor("strain_data", "frequency_domain_strain")
    delta_t = PropertyAccessor("strain_data", "delta_t")
    sampling_frequency = PropertyAccessor("strain_data", "sampling_frequency")
    duration = PropertyAccessor("strain_data", "duration")
    start_time = PropertyAccessor("strain_data", "start_time")
    segment = PropertyAccessor("strain_data", "segment")
    band = PropertyAccessor("strain_data", "band")
    nlength = PropertyAccessor("strain_data", "nlength")
    Nzeros = PropertyAccessor("strain_data", "Nzeros")
    crf0 = PropertyAccessor("strain_data", "crf0")
    mean = PropertyAccessor("strain_data", "mean")
    var = PropertyAccessor("strain_data", "var")

    start_time = PropertyAccessor("ephimeries", "start_time")
    duration = PropertyAccessor("ephimeries", "duration")
    delta_t = PropertyAccessor("ephimeries", "delta_t")
    segment = PropertyAccessor("ephimeries", "segment")
    band = PropertyAccessor("ephimeries", "band")
    ephemeries_dir = PropertyAccessor("ephimeries", "ephemeries_dir")
    DetSSB = PropertyAccessor("ephimeries", "DetSSB")
    nlength = PropertyAccessor("ephimeries", "nlength")
    phir = PropertyAccessor("ephimeries", "phir")
    epsm = PropertyAccessor("ephimeries", "epsm")

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
        delta_t,
        segment,
        band,
        ephimeries_dir,
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
        self.strain_data = StrainData(self.name, delta_t, segment, band, ephimeries_dir)

        self.ephimeries = Ephimeries(segment, band, delta_t, ephimeries_dir)

        self.ephimeries.load_detector_ephemeris(self.name)
        self.ephimeries.read_start_time(self.name)

        self.strain_data.load_time_domain_strain(self.name)
        self.strain_data.calculate_frequency_domain_strain()
        self.strain_data.start_time = self.ephimeries.start_time


def load_interferometer(filename, segment, band, delta_t, ephimeries_dir):
    """
    Load interferometer properties from a configuration file.

    Args:
        filename (str): Path to the configuration file.
    """

    parameters = dict()
    with open(filename, "r") as parameter_file:
        lines = parameter_file.readlines()
        for line in lines:
            if line[0] == "#" or line[0] == "\n":
                continue
            split_line = line.split("=")
            key = split_line[0].strip()
            value = eval("=".join(split_line[1:]))
            parameters[key] = value
    parameters["delta_t"] = delta_t
    parameters["segment"] = segment
    parameters["band"] = band
    parameters["ephimeries_dir"] = ephimeries_dir

    ifo = Interferometer(**parameters)
    return ifo


def get_empty_interferometer(name, segment, band, delta_t, ephimeries_dir="."):
    """
    Create an empty interferometer for a detector with it's properties
    """

    filename = os.path.join(
        os.path.dirname(__file__), "detectors", "{}.interferometer".format(name)
    )

    try:
        return load_interferometer(filename, segment, band, delta_t, ephimeries_dir)
    except OSError:
        raise ValueError("Interferometer {} not implemented".format(name))


class InterferometerList(list):
    """
    A list of Interferometer objects with additional utility methods.
    """

    def __init__(self, interferometers, segment, band, delta_t, ephimeries_dir="."):
        """
        Initialize the InterferometerList with a list of Interferometer objects
        """

        super(InterferometerList, self).__init__()
        if isinstance(interferometers, str):
            raise TypeError("Input must not be a string")
        for ifo in interferometers:
            if isinstance(ifo, str):
                ifo = get_empty_interferometer(
                    ifo, segment, band, delta_t, ephimeries_dir
                )
            if not isinstance(ifo, (Interferometer)):
                raise TypeError(
                    "Input list of interferometers are not all Interferometer objects"
                )
            else:
                self.append(ifo)
