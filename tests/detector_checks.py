import sys

sys.path.append("../")
import matplotlib.pyplot as plt
import numpy as np

import constants as cs
import detector

in_dir = "/work/chuck/virgo/O4/input_data_O4_G02/xdat_O4_6d/"
dt = 0.5
band = 17
segment = 1

num_days = 6
N = int((cs.C_SIDDAY * num_days) / dt)

det_h1 = detector.Detector("H1", in_dir, segment, band, N)
det_l1 = detector.Detector("L1", in_dir, segment, band, N)

print("phi_r of H1 detector for this segment and band is %s" % det_h1.phir)
print("phi_r of L1 detector for this segment and band is %s" % det_l1.phir)

print("epsm of H1 detector for this segment and band is %s" % det_h1.epsm)
print("epsm of L1 detector for this segment and band is %s" % det_l1.epsm)

det_ssb_h1 = np.reshape(det_h1.DetSSB, (N, 3))
det_ssb_h1_mag = np.sum(det_ssb_h1**2, axis=1)

det_ssb_l1 = np.reshape(det_l1.DetSSB, (N, 3))
det_ssb_l1_mag = np.sum(det_ssb_l1**2, axis=1)

print("The crf0 for the detector %s is %s" % (det_h1.name, det_h1.crf0))
print("The crf0 for the detector %s is %s" % (det_l1.name, det_l1.crf0))

print("The variance for the detector %s is %s" % (det_h1.name, det_h1.var))
print("The variance for the detector %s is %s" % (det_l1.name, det_l1.var))

start_time_h1 = det_h1.start_time
start_time_l1 = det_l1.start_time

if start_time_h1 != start_time_l1:
    print("The starting times for the two detectors are not the same!")
    exit(1)

start_time = start_time_h1

print("The start time for the test data is %s" % (start_time))

time_series = start_time + np.arange(0, N * dt, dt)

plt.figure(figsize=(10, 6))
plt.plot(time_series - time_series[0], det_ssb_h1_mag, label="H1", color="#ee0000")
plt.plot(time_series - time_series[0], det_ssb_l1_mag, label="L1", color="#4ba6ff")
plt.xlabel(r"Time - $t_{0}$ [s]")
plt.ylabel(r"Detector SSB")
plt.title(r"Detector SSB for O4 time segement 001 band 0017")
plt.legend()
plt.grid()
plt.savefig("./plots/DetSSB.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time_series - time_series[0], det_h1.data, label="H1", color="#ee0000")
plt.ylabel(r"Detector Data (H1)")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_series - time_series[0], det_l1.data, label="L1", color="#4ba6ff")
plt.xlabel(r"Time - $t_{0}$ [s]")
plt.ylabel(r"Detector Data (L1)")
plt.grid()
plt.legend()

plt.suptitle(r"Detector data for O4 time segement 001 band 0017")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("./plots/Detector-data.png", dpi=300)
plt.close()
