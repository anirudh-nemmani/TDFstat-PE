import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np

from TDFstat_PE import constants, utils
from TDFstat_PE.detector import interferometer as interferometer

strain_data_dir = "/work/chuck/virgo/O4/input_data_O4_G02/xdat_O4_6d/"
dt = 0.5
band = 17
segment = 1
overlap = 0.0625

detector_list = ["H1", "L1"]

check = interferometer.InterferometerList(detector_list, segment, band, dt, overlap)

check.load_detector_ephemeris(strain_data_dir)

check.load_time_domain_strain(strain_data_dir)
check.calculate_frequency_domain_strain()

for det in check:
    print("\n")
    print("Detector name: %s" % (det.name))
    print("phi_r = %s" % (det.ephemeris.phir))
    print("epsm = %s" % (det.ephemeris.epsm))
    print("crf0 = %s" % (det.strain_data.crf0))
    print("variance = %s" % (det.strain_data.var))
    print("\n")

time_series = (
    np.arange(
        0,
        (check[0].strain_data.nlength) * check[0].strain_data.delta_t,
        check[0].strain_data.delta_t,
    )
    + check[0].strain_data.start_time
)

colors = {"H1": "#ee0000", "L1": "#4ba6ff"}

plt.figure(figsize=(10, 6))
for det in check:
    det_ssb = det.ephemeris.DetSSB
    magnitude = np.sum(det_ssb**2, axis=1)
    plt.plot(
        time_series - time_series[0],
        magnitude,
        label=det.name,
        color=colors[det.name],
    )

plt.xlabel(r"Time $t - t_{0}$ [s]")
plt.ylabel(r"Detector SSB")
plt.title(r"Detector SSB for O4 time segement 001 band 0017")
plt.legend()
plt.grid()
plt.savefig("./plots/DetSSB.png", dpi=300)
plt.close()


print("The start time for the test data is %s" % (check[0].strain_data.start_time))

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(
    time_series - time_series[0],
    check[0].strain_data.time_domain_strain,
    label=check[0].name,
    color=colors[check[0].name],
)
plt.ylabel(r"Detector Data (%s)" % (check[0].name))
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(
    time_series - time_series[0],
    check[1].strain_data.time_domain_strain,
    label=check[1].name,
    color=colors[check[1].name],
)
plt.xlabel(r"Time $t - t_{0}$ [s]")
plt.ylabel(r"Detector Data (%s)" % (check[1].name))
plt.grid()
plt.legend()

plt.suptitle(r"Detector data for O4 time segement 001 band 0017")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("./plots/Time-domain-strain.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))

for det in check:
    plt.plot(
        det.strain_data.frequency_array,
        np.abs(det.strain_data.frequency_domain_strain),
        label=det.name,
        color=colors[det.name],
    )

plt.xlabel(r"Frequency [Hz]")
plt.xlim(
    check[0].strain_data.starting_frequency,
    check[0].strain_data.starting_frequency + check[0].strain_data.bandwidth,
)
plt.ylabel(r"|$\tilde{h}(f)$|")
plt.title(r"Frequency domain strain for O4 time segement 001 band 0017")
plt.legend()
plt.grid()
plt.savefig("./plots/Frequency-domain-strain.png", dpi=300)
plt.close()


### Gaussian generation checks
print("\n Generating Gaussian strain data for testing \n")
check = interferometer.InterferometerList(detector_list, segment, band, dt, overlap)
check.load_detector_ephemeris(strain_data_dir)
check.generate_gaussian_strain_data(
    check[0].ephemeris.start_time, check[0].ephemeris.duration, 1, seed=99
)
check.calculate_frequency_domain_strain()

for det in check:
    print("\n")
    print("Detector name: %s" % (det.name))
    print("crf0 (gaussian) = %s" % (det.strain_data.crf0))
    print("variance (gaussian) = %s" % (det.strain_data.var))
    print("\n")

print("The start time for the test data is %s" % (check[0].strain_data.start_time))

time_series = (
    np.arange(
        0,
        (check[0].strain_data.nlength) * check[0].strain_data.delta_t,
        check[0].strain_data.delta_t,
    )
    + check[0].strain_data.start_time
)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(
    time_series - time_series[0],
    check[0].strain_data.time_domain_strain,
    label=check[0].name,
    color=colors[check[0].name],
)
plt.ylabel(r"Detector Data (%s)" % (check[0].name))
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(
    time_series - time_series[0],
    check[1].strain_data.time_domain_strain,
    label=check[1].name,
    color=colors[check[1].name],
)
plt.xlabel(r"Time $t - t_{0}$ [s]")
plt.ylabel(r"Detector Data (%s)" % (check[1].name))
plt.grid()
plt.legend()

plt.suptitle(r"Detector data for O4 time segement 001 band 0017")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("./plots/Time-domain-Gaussian-strain.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
for det in check:
    plt.plot(
        det.strain_data.frequency_array,
        np.abs(det.strain_data.frequency_domain_strain),
        label=det.name,
        color=colors[det.name],
    )
plt.xlabel(r"Frequency [Hz]")
plt.xlim(
    check[0].strain_data.starting_frequency,
    check[0].strain_data.starting_frequency + check[0].strain_data.bandwidth,
)
plt.ylabel(r"|$\tilde{h}(f)$|")
plt.title(r"Frequency domain Gaussian noise for O4 time segement 001 band 0017")
plt.legend()
plt.grid()
plt.savefig("./plots/Frequency-domain-Gaussian-strain.png", dpi=300)
plt.close()
