import struct

import numpy as np


class Detector:
    """
    Class to handle detector data and ephemeris information.

    Attributes:
        name (str): Name of the detector.
        path (str): Input directory path.
        seg (int): Segment number.
        band (int): Frequency band number.
        N (int): Number of data points.
    """

    def __init__(self, name, in_dir, seg, band, N):
        self.name = name  # Detector name
        self.path = in_dir  # Input directory

        self.seg = seg  # Segment number
        self.band = band  # Frequency band number
        self.N = N  # Number of data points

        self.ephi = 0.0  # Geographical latitude phi in radians
        self.elam = 0.0  # Geographical longitude in radians
        self.eheight = 0.0  # Height h above the Earth ellipsoid in meters
        self.egam = 0.0  # Orientation of the detector gamma

        self.DetSSB = np.zeros(3 * N, dtype=np.float64)  # Ephemeris of the detector
        self.phir = 0.0
        self.epsm = 0.0

        self.data = np.zeros(N, dtype=np.float64)  # Time-domain data
        self.Nzeros = 0  # Number of 0s in the data
        self.crf0 = 0.0  # Number of 0s as: N / (N - Nzeros)
        self.mean = 0.0  # Mean of the detector data
        self.var = 0.0  # Variance of the detector data

        self.start_time = 0.0  # Start time of the data segment

        self.get_DetSSB()
        self.add_data()
        self.get_start_time()

    def get_DetSSB(self):
        """
        Load the ephemeris data for a given detector from the DetSSB file.
        """
        filename = f"{self.path}/{self.seg:03d}/{self.name}/DetSSB.bin"

        try:
            with open(filename, "rb") as data:
                buffer = data.read()
                self.DetSSB = np.frombuffer(buffer[: 3 * self.N * 8], dtype=np.float64)
                self.phir, self.epsm = struct.unpack(
                    "dd", buffer[3 * self.N * 8 : 3 * self.N * 8 + 16]
                )
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
            return

    def add_data(self):
        """
        Load the time-domain data from the xDat file for a given detector
        at a given band and segment. Also computes the variance of the data,
        taking into account any null values present.
        """
        xdatname = (
            f"{self.path}/{self.seg:03d}/{self.name}/"
            f"xdat_{self.seg:03d}_{self.band:04d}.bin"
        )

        try:
            with open(xdatname, "rb") as data:
                self.data = np.frombuffer(data.read(self.N * 4), dtype=np.float32)
                print(
                    f"Loaded data for detector {self.name}, band {self.band} in Float32 format, if \n using O3 data please make sure the read data is in Float64"
                )
        except FileNotFoundError:
            print(f"Error: {xdatname} not found.")
            exit(1)

        # Checking for null values in the data
        self.Nzeros = np.count_nonzero(self.data == 0)

        # Factor N / (N - Nzeros) to account for null values in the data
        if self.Nzeros < self.N:
            self.crf0 = self.N / (self.N - self.Nzeros)
        else:
            print("Warning: All data points are zero. Exiting.")
            exit(1)

        # Estimation of the variance for each detector
        self.mean = np.mean(self.data)
        self.var = self.crf0 * np.mean((self.data - self.mean) ** 2)

    def get_start_time(self):
        """
        Load the start time of the data segment from the starting_date file.
        """
        start_time_filename = f"{self.path}/{self.seg:03d}/{self.name}/starting_date"

        try:
            with open(start_time_filename, "r") as data:
                self.start_time = float(data.read().strip())
        except FileNotFoundError:
            print(f"Error: {start_time_filename} not found.")
            return
