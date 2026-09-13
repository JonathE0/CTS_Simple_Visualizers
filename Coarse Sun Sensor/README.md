# Coarse Sun Sensor Visualizer

An interactive 1U CubeSat visualization showing how six ideal cosine Sun sensors respond as the spacecraft rotates. Adjust the Sun direction, spacecraft attitude, spin axes, and light intensity to explore the sensor currents and estimated Sun direction.

## Requirements

- Git (Git Bash on Windows)
- Python 3.10 or newer

No additional Python packages are required.

## Download and run

Open Git Bash in the directory where you want to download the repository, then run:

```bash
git clone https://github.com/JonathE0/CTS_Simple_Visualizers.git
cd "CTS_Simple_Visualizers/Coarse Sun Sensor"
python Cubesat_sun_sensor.py
```

If your system uses `python3`, use `python3 Cubesat_sun_sensor.py` for the last command. Folder and filename capitalization should match the commands above; the quotes around the folder path are needed because it contains spaces.

Your browser should open automatically. If it does not, open [the local simulator](http://127.0.0.1:8000).

Keep the terminal open while using the simulator. Press **Ctrl+C** in the terminal to stop it.

## Run it again

You only need to clone the repository once. For later sessions, open Git Bash in the `Coarse Sun Sensor` folder and run:

```bash
python Cubesat_sun_sensor.py
```

## Troubleshooting

If Python is not found, install Python 3.10 or newer, enable its PATH option on Windows, and reopen Git Bash.

If port 8000 is already in use, choose another port:

```bash
python Cubesat_sun_sensor.py --port 8080
```

Then open [the simulator on port 8080](http://127.0.0.1:8080). To prevent the browser from opening automatically, add `--no-browser`. Run with `--help` to see all options.

## Model

Each face uses the ideal response `I = K * max(0, cos(phi))`, where `phi` is the angle between its outward normal and the direction to the Sun. Opposite-face current differences recover the Sun direction when the common light intensity is positive.

This is an educational visualization. It does not model noise, eclipse, albedo, saturation, or orbital dynamics. A Sun direction alone does not determine full spacecraft attitude.
