import configparser

import numpy as np

from . import constants as cs


class ReadIni:
    def __init__(self, ini_path):
        self.ini_path = ini_path
        self.params = {}
        self._read_ini()

    def _read_ini(self):
        """
        Read parameters from an INI file and parse them as a dictionary.
        """
        config = configparser.ConfigParser()
        config.read(self.ini_path)

        general_params = config["general"]
        settings = config["settings"]
        signal_params = config["signal"]

        # Parse general parameters
        self.params["output_dir"] = general_params.get("output_dir", "./")

        # Parse settings parameters
        self.params["detectors"] = settings.get("detectors", "H1,L1").split(",")
        self.params["indir"] = settings.get(
            "indir", "/work/chuck/virgo/O4/input_data_O4_G02/xdat_O4_6d"
        )
        if "bandwidth" in settings:
            self.params["bandwidth"] = float(settings.get("bandwidth", 1.0))
            self.params["dt"] = 1.0 / (2.0 * self.params["bandwidth"])
        elif "delta_t" in settings:
            self.params["dt"] = float(settings.get("delta_t", 0.5))
            self.params["bandwidth"] = 1.0 / (2.0 * self.params["dt"])
        else:
            raise ValueError(
                "INI file must contain either 'bandwidth' or 'delta_t' in the [settings] section"
            )
        if "ov" in settings:
            self.params["ov"] = float(settings.get("ov", 4.0))
            self.params["overlap"] = 2 ** (-self.params["ov"])
        elif "overlap" in settings:
            self.params["overlap"] = float(settings.get("overlap", 0.0625))
            self.params["ov"] = -np.log2(self.params["overlap"])
        else:
            raise ValueError(
                "INI file must contain either 'ov' or 'overlap' in the [settings] section"
            )
        self.params["duration"] = int(
            settings.get("duration", 6)
        )  # Duration in days, should be considered to in sidereal days only
        self.params["duration"] = (
            self.params["duration"] * cs.C_SIDDAY
        )  # Convert to seconds
        self.params["N"] = int(self.params["duration"] / self.params["dt"])

        self.params["seg"] = int(settings.get("segment", 1))
        self.params["band"] = int(settings.get("band", 17))

        # Parse cw signal parameters
        self.params["amplitude"] = float(
            signal_params.get("amplitude", 6.262497962511783e-5)
        )
        self.params["frequency"] = float(signal_params.get("frequency", 26.3589129))
        self.params["spindown"] = float(signal_params.get("spindown", -8.50e-11))
        self.params["ra"] = float(signal_params.get("ra", 3.86687548714555))
        self.params["dec"] = float(signal_params.get("dec", 0.74835013347064))
        self.params["inclination"] = float(
            signal_params.get("inclination", 2.984905359501588)
        )
        self.params["polarization"] = float(
            signal_params.get("polarization", 0.61465097870534)
        )
        self.params["wobble_angle"] = float(
            signal_params.get("wobble_angle", np.pi / 2)
        )
        self.params["phase"] = float(signal_params.get("phase", 0.05813090969327))

        if float(signal_params.get("reference_time", 930582085)) != 0:
            self.params["reference_time"] = float(
                signal_params.get("reference_time", 930582085)
            )


class PropertyAccessor(object):
    """
    Copied from Bilby : https://github.com/bilby-dev/bilby/blob/main/bilby/core/utils/introspection.py#L122-L151

    Generic descriptor class that allows handy access of properties without long
    boilerplate code. The properties of Interferometer are defined as instances
    of this class.

    This avoids lengthy code like

    .. code-block:: python

        @property
        def length(self):
            return self.geometry.length

        @length_setter
        def length(self, length)
            self.geometry.length = length

    in the Interferometer class
    """

    def __init__(self, container_instance_name, property_name):
        self.property_name = property_name
        self.container_instance_name = container_instance_name

    def __get__(self, instance, owner):
        return getattr(
            getattr(instance, self.container_instance_name), self.property_name
        )

    def __set__(self, instance, value):
        setattr(
            getattr(instance, self.container_instance_name), self.property_name, value
        )
        setattr(
            getattr(instance, self.container_instance_name), self.property_name, value
        )
