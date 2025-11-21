import numpy as np
import detector
import configparser
import constants


class Signal:


    def __init__(self, ini_path: str):
        self.ini_path = ini_path
        self._read_ini()
        self._generate_time_array()
        self._generate_signal()
        self.omega_r = constants.C_OMEGA_R
        

    
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


    def sampling_time(self):

        """
        dt in TDFstat
        """

        return 1.0/(2.0 * self.bandwidth)
    
    # Amplitude modulation function ( a replica of modvir in settings.c)

    def amplitude_modulation(self):

        return


    def OMEGA_S(self):
        """
        Dimensionless angular frequency 
        """

        return 2*np.pi*self.frequency * self.sampling_time
        
        
    def four_amplitudes(self):

        #To be cross-verified with TDFstat paper 1 (eq.32,33,34,35)


        term1 = np.sin(self.wobble_angle)*np.sin(self.wobble_angle)
        term2 = 0.5*(1+np.cos(self.inclination)*np.cos(self.inclination))
        
        psi = self.polarization
        phi0 = self.phase 
        iota = self.inclination

        A21 = term1*(term2*np.cos(2*psi)*np.cos(2*phi0) -np.cos(iota)*np.sin(2*psi)*np.sin(2*phi0) )
        A22  = term1*(term2 * np.sin(2*psi)*np.cos(2*phi0) + np.cos(iota)*np.cos(2*psi)*np.sin(2*phi0))

        A23 = term1*(-term2*np.cos(2*psi)*np.sin(2*phi0)- np.cos(iota)*np.sin(2*psi)*np.cos(2*phi0) )
        A24 = term1*(-term2*np.sin(2*psi)*np.sin(2*phi0)+np.cos(iota)*np.cos(2*psi)*np.cos(2*phi0))

        return A21,A22,A23,A24







