import struct

import numpy as np

import constants as cs



class Detector:
    def __init__(self, name, xdatname, N):
        self.name = name
        self.xdatname = xdatname

        self.ephi = 0.0  # Geographical latitude phi in radians
        self.elam = 0.0  # Geographical longitude in radians
        self.eheight = 0.0  # Height h above the Earth ellipsoid in meters
        self.egam = 0.0  # Orientation of the detector gamma

        self.sig = Signal(N)
        self.amod = Amp_mod_coeff()


class Amp_mod_coeff:
    def __init__(self):
        self.c1 = 0.0
        self.c2 = 0.0
        self.c3 = 0.0
        self.c4 = 0.0
        self.c5 = 0.0
        self.c6 = 0.0
        self.c7 = 0.0
        self.c8 = 0.0
        self.c9 = 0.0


class Signal:
    def __init__(self, N):
        self.xDat = np.zeros(N, dtype=np.float64)
        self.DetSSB = np.zeros(3 * N, dtype=np.float64)  # Ephimeries of the detector
        self.aa = np.zeros(N, dtype=np.float64)  # Amplitude modulation a(t)
        self.bb = np.zeros(N, dtype=np.float64)  # Amplitude modulation b(t)
        self.shft = np.zeros(N, dtype=np.float64)  # Resampling
        self.shftf = np.zeros(N, dtype=np.float64)  # Time shifting

        self.Nzeros = 0

        self.epsm = 0.0
        self.phir = 0.0
        self.sepsm = 0.0  # sin(epsm)
        self.cepsm = 0.0  # cos(epsm)
        self.sphir = 0.0  # sin(phi_r)
        self.cphir = 0.0  # cos(phi_r)

        self.crf0 = 0.0  # number of 0s as: N/(N-Nzeros)
        self.sig2 = 0.0  # variance of the signal

        self.xDatma = np.zeros(N, dtype=np.complex128)
        self.xDatmb = np.zeros(N, dtype=np.complex128)


class Settings:
    def __init__(self, N, nifo, omr, seg, band):
        self.N = N  # number of data points
        self.nifo = nifo  # number of detectors
        self.omr = omr  # C_OMEGA_R * dt (dimensionless Earth's angular frequency)
        self.ifo = [None] * nifo  # List of the detectors
        self.sepsm = 0.0  # sin(epsm)
        self.cepsm = 0.0  # cos(epsm)

        self.seg = seg  # segment number
        self.band = band  # frequency band


class Options:
    def __init__(self, indir, seg, mods=None):
        self.indir = indir  # input directory
        self.seg = seg  # segment number
        self.mods = mods if mods else []  # If there are any mods included


class AuxiliaryArrays:
    def __init__(self, N):
        self.t2 = np.zeros(N, dtype=np.float64)  # time^2
        self.cosmodf = np.zeros(N, dtype=np.float64)  # Earth position cosine
        self.sinmodf = np.zeros(N, dtype=np.float64)  # Earth position sine
        self.injection = np.zeros(
            10, dtype=np.float64
        )  # Injection array : noi, reference time, h0, f0, fdot, ra, dec, iota, psi, phi0


class ArrayInitializer:
    def __init__(self, sett: Settings, opts: Options, aux_arr: AuxiliaryArrays):
        self.sett = sett
        self.opts = opts
        self.aux_arr = aux_arr

    def init_arrays(self):
        for i in range(self.sett.nifo):
            print(self.sett.ifo[i].name)

    # def init_arrays(self):
    #     for i in range(self.sett.nifo):
    #         ifo = self.sett.ifo[i]

    #         # Input time-domain data handling

    #         try:
    #             with open(ifo.xdatname, "rb") as data:
    #                     ifo.sig.xDat = np.frombuffer(
    #                         data.read(self.sett.N * 8), dtype=np.float64
    #                     )
    #                 else:
    #                     ifo.sig.xDat = np.frombuffer(
    #                         data.read(self.sett.N * 4), dtype=np.float32
    #                     ).astype(np.float64)
    #         except FileNotFoundError:
    #             print(f"Error: {ifo.xdatname} not found.")
    #             exit(1)

    #         # Checking for null values in the data
    #         ifo.sig.Nzeros = np.sum(ifo.sig.xDat == 0)

    #         # Factor N/(N - Nzeros) to account for null values in the data
    #         ifo.sig.crf0 = (
    #             self.sett.N / (self.sett.N - ifo.sig.Nzeros)
    #             if ifo.sig.Nzeros < self.sett.N
    #             else float("inf")
    #         )

    #         # Estimation of the variance for each detector
    #         mean_xDat = np.mean(ifo.sig.xDat)
    #         ifo.sig.sig2 = ifo.sig.crf0 * np.mean((ifo.sig.xDat - mean_xDat) ** 2)

    #         # Ephemeris file handling
    #         filename = f"{self.opts.indir}/{self.opts.seg:03d}/{ifo.name}/DetSSB.bin"

    #         try:
    #             with open(filename, "rb") as data:
    #                 ifo.sig.DetSSB = np.frombuffer(
    #                     data.read(3 * self.sett.N * 8), dtype=np.float64
    #                 )
    #                 ifo.sig.phir = struct.unpack("d", data.read(8))[0]
    #                 ifo.sig.epsm = struct.unpack("d", data.read(8))[0]
    #                 print(f"Using {filename} as detector {ifo.name} ephemerids...")
    #         except FileNotFoundError:
    #             print(f"Error: {filename} not found.")
    #             return

    #         # sincos
    #         ifo.sig.sphir = np.sin(ifo.sig.phir)
    #         ifo.sig.cphir = np.cos(ifo.sig.phir)
    #         ifo.sig.sepsm = np.sin(ifo.sig.epsm)
    #         ifo.sig.cepsm = np.cos(ifo.sig.epsm)

    #         self.sett.sepsm = ifo.sig.sepsm
    #         self.sett.cepsm = ifo.sig.cepsm

    #         ifo.sig.xDatma = np.zeros(self.sett.N, dtype=np.complex128)
    #         ifo.sig.xDatmb = np.zeros(self.sett.N, dtype=np.complex128)

    #         ifo.sig.aa = np.zeros(self.sett.N, dtype=np.float64)
    #         ifo.sig.bb = np.zeros(self.sett.N, dtype=np.float64)

    #         ifo.sig.shft = np.zeros(self.sett.N, dtype=np.float64)
    #         ifo.sig.shftf = np.zeros(self.sett.N, dtype=np.float64)

    #     # Check if the ephemerids have the same epsm parameter
    #     for i in range(1, self.sett.nifo):
    #         if self.sett.ifo[i - 1].sig.sepsm != self.sett.ifo[i].sig.sepsm:
    #             print(
    #                 f"The parameter epsm (DetSSB.bin) differs for detectors {self.sett.ifo[i - 1].name} and {self.sett.ifo[i].name}. Aborting..."
    #             )
    #             exit(1)

    #     # if all is well with epsm, take the first value
    #     self.sett.sepsm = self.sett.ifo[0].sig.sepsm
    #     self.sett.cepsm = self.sett.ifo[0].sig.cepsm

    #     # Auxiliary arrays, Earth's rotation
    #     indices = np.arange(self.sett.N)
    #     omrt = self.sett.omr * indices  # Earth angular velocity * dt * i
    #     self.aux_arr.t2 = indices**2
    #     self.aux_arr.t2 = indices ** 2
    #     self.aux_arr.cosmodf = np.cos(omrt)
    #     self.aux_arr.sinmodf = np.sin(omrt)
