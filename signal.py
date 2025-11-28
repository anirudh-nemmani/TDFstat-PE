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
        self.phir = #to be taken from detector class
        self.gamma = #to be taken from detector class
        self.lambda = # to be taken from detector class
        
        

    
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
    
    def nSource(self):

        n1 = np.cos(self.ra)*np.cos(self.dec)
        n2 = np.cos()
        n3 = np.sin(self.dec)

        return [n1,n2,n3]


    
    # Amplitude modulation function ( a replica of modvir in settings.c)

    def amplitude_modulation(self):

        modulation = self.ra-self.phir-self.omega_r*t
        a1  = (1/16)*np.sin(2*self.gamma)*(3 - np.cos(2*self.lambda))*(3- np.cos(2*self.dec))*np.cos(2*(modulation))
        a2 = -0.25*np.cos(2*self.gamma)*np.sin(self.lambda)*(3-np.cos(2*self.dec))*np.sin(2*(modulation))
        a3 = 0.25*np.sin(2*self.gamma)*np.sin(2*self.lambda)*np.sin(2*self.dec)*np.cos(modulation)
        a4 = -0.5*np.cos(2*self.gamma)*np.cos(self.lambda)*np.sin(2*self.dec)*np.sin(modulation)
        a5 = 0.75*np.sin(2*self.gamma)*(np.cos(self.lambda))**2 * (np.cos(self.dec))**2

        a  = a1 +a2 +a3 +a4 +a5 
        
        b1 = np.cos(2*self.gamma)*np.sin(self.lambda)*np.sin(self.dec)*np.cos(2*modulation)
        b2 = 0.25*np.sin(2*self.gamma)*(3 - np.cos(2*self.lambda))*np.sin(self.dec)*np.sin(2*modulation)
        b3 = np.cos(2*self.gamma)*np.cos(self.lambda)*np.cos(self.dec)*np.cos(modulation)
        b4 = 0.25*np.sin(2*self.gamma)*np.sin(2*self.lambda)*np.cos(self.self.dec)*np.sin(modulation)

        b = b1 + b2 + b3 + b4 


        return a, b


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
    
    def phase_function(self):
         
        return 
    def signal(self):

        A21,A22,A23,A24 = self.four_amplitudes()
        a,b = self.amplitude_modulation()

        return (a*A21 + b*A22 )*np.cos(self.phase_function()) + (a*A23 + b*A24)*np.sin(self.phase_function())










