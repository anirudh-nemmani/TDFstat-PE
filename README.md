# TDFstat-PE

This repository is under work in progress......

The repository is aimed to utilise parameter estimation methods from [Bilby](https://github.com/bilby-dev/bilby) to estimate parameters of continuous gravitational wave sources.

To do this, we use the `add_signal` functionality from [TDFstat](https://github.com/Polgraw/TDFstat) package and convert it to a python class to generate signals at mutliple parameters.

We also initialise the detectors ourselves to read the ephemeris files.

Checklist of functionalities to be added :

- [x] Create a detector class to read the ephemeris files, data files and starting time.
- [ ] Create a signal class to generate signals at multiple parameters.
- [ ] Integrate Bilby parameter estimation methods to estimate parameters of the generated signals.
- [ ] Have a consistent INI parser file to read all the necessary inputs.
- [ ] Test the package on simulated data.
- [ ] Documentation for the package and create a tutorial.

All the class tests will be included in the tests folder.

If anyone wants to contribute to this package, please feel free to reach out to the maintainers.

Note : Depending on the computational costs for paramter estimation, we will require contributors in optimising the code in CPU as well as GPU platforms.

Current maintainers/contributors :

[Anirudh Nemmani](https://github.com/anirudh-nemmani)

[Sreekanth Harikumar](https://github.com/hsreekanth15)
