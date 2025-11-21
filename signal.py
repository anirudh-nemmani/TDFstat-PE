import numpy as np
import detector
import configparser


class Signal:


    def __init__(self, ini_path: str):
        self.ini_path = ini_path
        self._read_ini()
        self._generate_time_array()
        self._generate_signal()

    
    def _read_ini(self):
        config = configparser.ConfigParser()
        config.read(self.ini_path)

        if "signal" not in config:
            raise ValueError("INI file must contain a [signal] section")

        signal_params = config["signal"]
        search_params = config["search"]

        # Parse cw signal parameters
        self.amplitude = float(signal_params.get("amplitude", 1.0))
        self.frequency = float(signal_params.get("frequency", 1.0))
        self.frequency_dot = float(signal_params.get("frequency_dot", 1.0))
        self.ra = float(signal_params.get("ra",1.0))
        self.dec = float(signal_params.get("dec",1.0))
        self.inclination = float(signal_params.get("inclination",1.0))
        self.polarization = float(signal_params.get("polarization",1.0))
        self.wobble_angle = float(signal_params.get("wobble_angle",np.pi/2))
        self.phase = float(signal_params.get("phase", 0.0))

        self.bandwidth = float(search_params.get("bandwidth", 1.0))
        self.ov = float(search_params.get("ov",4.0))
        self.band = float(search_params.get("band", 1234))

        self.sampling_rate = float(signal_params.get("sampling_rate", 44100))
        self.duration = float(section.get("duration", 1.0))

    def four_amplitudes(self):

        return A11,A12,A21,A22




