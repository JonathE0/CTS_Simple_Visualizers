# Manual Throttle Control

An interactive 1U CubeSat attitude simulator with three ideal reaction wheels. Use the vertical momentum throttles to stop a random tumble or point the satellite's +Z axis toward the Sun. Includes the light-blue visual theme.

## Download and run

Requires Python 3.10 or newer. No additional Python packages are needed.

```bash
git clone https://github.com/JonathE0/CTS_Simple_Visualizers.git
cd "CTS_Simple_Visualizers/Manual Throttle Control"
python cubesat_attitude.py
```

If you already cloned this repository, run `git pull` from that checkout instead of cloning again. On systems that use `python3`, substitute it for `python`.

The browser opens automatically at http://127.0.0.1:8002. Keep the terminal open; press Ctrl+C to stop the server. For another port or to avoid opening a browser automatically:

```bash
python cubesat_attitude.py --port 8080 --no-browser
```

This is a locally served Python application; opening the Python file on GitHub does not run the simulator.

## Controls

- Adjust Sun azimuth and elevation to change the desired pointing direction.
- Set each wheel's target momentum with its vertical throttle, numbered notches, +/- buttons, or numeric field.
- Match each throttle to its own green suggested-target band. Guidance is advisory; it does not move the throttles.
- Choose a target refresh interval from 0.1 to 1.0 seconds. Physics updates independently of that display setting.
- Select **Random tumble** to start a detumbling exercise. Use **Detumble only**, then switch to **Point +Z at Sun** when ready.
- Drag the scene to change the camera view. Reset returns the spacecraft to rest with +Z toward the initial Sun direction.

## Model

The 0.1 m cube has an effective carrier mass of 1 kg and inertia of mL^2/6 per axis. Three orthogonal ideal wheels are located at the center of mass, each with axial inertia 1e-5 kg m^2.

Throttles command wheel momentum, not instantaneous torque or RPM. A proportional motor servo approaches each target with a 1-second time constant and a torque limit of +/-0.1 mN m. The displayed relative wheel RPM accounts for both wheel momentum and spacecraft body rate.

Motor torque on the satellite is equal and opposite to motor torque on the wheel. The rigid-body model also includes gyroscopic coupling and conserves total inertial angular momentum without external torque. After detumbling, wheels may need to keep spinning; centering the throttles can restart body rotation.

This educational model assumes exact state information and excludes disturbances, sensor noise, friction, and physical wheel momentum/speed limits. The throttle range is a command limit, not a wheel hardware capacity. Sun pointing does not specify the final roll angle about +Z. There is no automatic satellite pointing controller.

## Verification

From this folder:

```bash
python -m unittest test_cubesat_attitude
```
