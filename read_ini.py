import configparser

import numpy as np


class read_ini:
    def __init__(self, ini_path):
        self.ini_path = ini_path
        self._read_ini()

    def _read_ini(self):
        """
        Read parameters from an INI file and set them as class attributes.
        """
        config = configparser.ConfigParser()
        config.read(self.ini_path)

        general_params = config["general"]
        settings = config["settings"]
        signal_params = config["signal"]

        # Parse general parameters
        self.output_dir = general_params.get("output_dir", "./")

        # Parse settings parameters
        self.detectors = settings.get("detectors", "H1,L1").split(",")
        self.indir = settings.get(
            "indir", "/work/chuck/virgo/O4/input_data_O4_G02/xdat_O4_6d"
        )
        if "bandwidth" in settings:
            self.bandwidth = float(settings.get("bandwidth", 1.0))
            self.dt = 1.0 / (2.0 * self.bandwidth)
        elif "delta_t" in settings:
            self.dt = float(settings.get("delta_t", 0.5))
            self.bandwidth = 1.0 / (2.0 * self.dt)
        else:
            raise ValueError(
                "INI file must contain either 'bandwidth' or 'delta_t' in the [settings] section"
            )
        if "ov" in settings:
            self.ov = float(settings.get("ov", 4.0))
            self.overlap = 2 ** (-self.ov)
        elif "overlap" in settings:
            self.overlap = float(settings.get("overlap", 0.0625))
            self.ov = -np.log2(self.overlap)
        else:
            raise ValueError(
                "INI file must contain either 'ov' or 'overlap' in the [settings] section"
            )
        self.seg = int(settings.get("segment", 1))
        self.band = int(settings.get("band", 17))

        # Parse cw signal parameters
        self.amplitude = float(signal_params.get("amplitude", 6.262497962511783e-5))
        self.frequency = float(signal_params.get("frequency", 26.3589129))
        self.spindown = float(signal_params.get("spindown", -8.50e-11))
        self.ra = float(signal_params.get("ra", 3.86687548714555))
        self.dec = float(signal_params.get("dec", 0.74835013347064))
        self.inclination = float(signal_params.get("inclination", 2.984905359501588))
        self.polarization = float(signal_params.get("polarization", 0.61465097870534))
        self.wobble_angle = float(signal_params.get("wobble_angle", np.pi / 2))
        self.phase = float(signal_params.get("phase", 0.05813090969327))

        if float(signal_params.get("reference_time", 930582085)) != 0:
            self.reference_time = float(signal_params.get("reference_time", 930582085))
