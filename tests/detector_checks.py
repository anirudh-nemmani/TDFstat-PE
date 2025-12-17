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

detector_list = ["H1", "L1"]

check = interferometer.InterferometerList(
    detector_list, segment, band, dt, strain_data_dir
)

for det in check:
    print("\n")
    print("Detector name: %s" % (det.name))
    print("phi_r = %s" % (det.ephimeries.phir))
    print("epsm = %s" % (det.ephimeries.epsm))
    print("crf0 = %s" % (det.strain_data.crf0))
    print("variance = %s" % (det.strain_data.var))
    print("\n")

time_series = np.arange(
    0,
    (check[0].strain_data.nlength) * check[0].strain_data.delta_t,
    check[0].strain_data.delta_t,
)

colors = {"H1": "#ee0000", "L1": "#4ba6ff"}

plt.figure(figsize=(10, 6))
for det in check:
    det_ssb = det.ephimeries.DetSSB
    magnitude = np.sum(det_ssb**2, axis=1)
    plt.plot(time_series, magnitude, label=det.name, color=colors[det.name])
plt.xlabel(r"Time$ [s]")
plt.ylabel(r"Detector SSB")
plt.title(r"Detector SSB for O4 time segement 001 band 0017")
plt.legend()
plt.grid()
plt.savefig("./plots/DetSSB.png", dpi=300)
plt.close()


print("The start time for the test data is %s" % (check[0].strain_data.start_time))

time_series = check[0].strain_data.start_time + np.arange(
    0,
    check[0].strain_data.nlength * check[0].strain_data.delta_t,
    check[0].strain_data.delta_t,
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
plt.xlabel(r"Time - $t_{0}$ [s]")
plt.ylabel(r"Detector Data (%s)" % (check[1].name))
plt.grid()
plt.legend()

plt.suptitle(r"Detector data for O4 time segement 001 band 0017")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("./plots/Detector-data.png", dpi=300)
plt.close()
