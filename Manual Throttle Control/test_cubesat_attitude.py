import math
import random
import unittest

try:
    import cubesat_attitude as model
except ImportError:
    model = None


class AttitudeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(model, 'Attitude model is not implemented')

    def advance(self, state, torque, seconds):
        for _ in range(round(seconds / .05)):
            state = model.advance(state, torque, .05)
        return state

    def test_initial_rest_and_alignment(self):
        state = self.advance(model.initial_state(), [0, 0, 0], 1)
        self.assertEqual(state['omega'], [0, 0, 0])
        self.assertEqual(state['q'], [1, 0, 0, 0])
        self.assertAlmostEqual(model.snapshot(state)['error_deg'], 0)

    def test_single_axis_motor_torque_and_analytic_angle(self):
        state = self.advance(model.initial_state(), [.0001, 0, 0], 1)
        self.assertAlmostEqual(state['omega'][0], -.06, places=10)
        self.assertAlmostEqual(state['h'][0], .0001, places=12)
        # alpha=-0.06 rad/s², theta=0.5*alpha*t²=-0.03 rad.
        self.assertAlmostEqual(state['q'][0], math.cos(.015), places=10)
        self.assertAlmostEqual(state['q'][1], -math.sin(.015), places=10)

    def test_zero_torque_coasts_and_reverse_torque_brakes(self):
        state = self.advance(model.initial_state(), [.0001, 0, 0], 1)
        coast = self.advance(state, [0, 0, 0], 1)
        self.assertAlmostEqual(coast['omega'][0], -.06, places=10)
        stopped = self.advance(coast, [-.0001, 0, 0], 1)
        self.assertAlmostEqual(stopped['omega'][0], 0, places=10)
        self.assertAlmostEqual(stopped['h'][0], 0, places=12)

    def test_inertial_total_momentum_is_conserved(self):
        state = model.initial_state()
        state['omega'] = [.2, -.1, .3]
        state['h'] = [.001, -.002, .0005]
        expected = [.001 + .2/600, -.002 - .1/600, .0005 + .3/600]
        state = self.advance(state, [.00007, -.00004, .00009], 5)
        for actual, value in zip(model.snapshot(state)['momentum_world'], expected):
            self.assertAlmostEqual(actual, value, places=10)
        self.assertAlmostEqual(math.hypot(*state['q']), 1, places=12)

    def test_cross_product_sign_and_sun_does_not_change_state(self):
        state = model.initial_state()
        result = model.snapshot(state, azimuth=0, elevation=0)
        self.assertEqual(result['cross_body'], [0, 1, 0])
        self.assertAlmostEqual(result['error_deg'], 90)
        self.assertEqual(state['q'], [1, 0, 0, 0])

    def test_antiparallel_is_not_reported_as_aligned(self):
        result = model.snapshot(model.initial_state(), elevation=-90)
        self.assertAlmostEqual(result['error_deg'], 180)
        self.assertTrue(result['antiparallel'])
        self.assertLess(math.hypot(*result['cross_body']), 1e-12)

    def test_torque_pair_and_relative_wheel_speed(self):
        state = self.advance(model.initial_state(), [.0001, 0, 0], 1)
        result = model.snapshot(state, torque=[.0001, 0, 0])
        self.assertEqual(result['satellite_torque'], [-.0001, 0, 0])
        self.assertAlmostEqual(result['wheel_rpm'][0], 10.06*60/(2*math.pi), places=8)

    def test_invalid_inputs_and_zero_dt(self):
        state = model.initial_state()
        self.assertEqual(model.advance(state, [0, 0, 0], 0), state)
        for torque, dt in [([.1,0,0], .05), ([0,0,0], -.1), ([0,0,0], 2), ([math.nan,0,0], .05)]:
            with self.assertRaises(ValueError):
                model.advance(state, torque, dt)
        state['q'] = [0,0,0,0]
        with self.assertRaises(ValueError):
            model.advance(state, [0,0,0], .05)

    def test_momentum_throttle_ramps_and_holds(self):
        state = model.initial_state()
        state, torque = model.advance_targets(state, [.0003, 0, 0], .05)
        self.assertAlmostEqual(state['h'][0], .000005, places=12)
        self.assertLessEqual(abs(torque[0]), .0001)
        for _ in range(300):
            state, torque = model.advance_targets(state, [.0003, 0, 0], .05)
        self.assertAlmostEqual(state['h'][0], .0003, places=9)
        self.assertLess(abs(torque[0]), 1e-9)
        self.assertLess(state['omega'][0], 0)

    def test_center_throttle_brakes_without_overshooting(self):
        state = self.advance(model.initial_state(), [.0001, 0, 0], 1)
        for _ in range(300):
            state, torque = model.advance_targets(state, [0,0,0], .05)
            self.assertGreaterEqual(state['h'][0], 0)
        self.assertAlmostEqual(state['omega'][0], 0, places=7)

    def test_guidance_points_correct_way_and_damps_rotation(self):
        state = model.initial_state()
        result = model.control_snapshot(state, [0,0,0], azimuth=0, elevation=0)
        self.assertLess(result['suggested_target'][1], 0)
        self.assertEqual(result['suggested_axis'], 1)
        state['omega']=[0,0,.1]
        result = model.control_snapshot(state, [0,0,0])
        self.assertGreater(result['suggested_target'][2], 0)

    def test_momentum_targets_are_bounded(self):
        with self.assertRaises(ValueError):
            model.advance_targets(model.initial_state(), [.001,0,0], .05)

    def test_guidance_all_axes_and_lever_detents(self):
        for azimuth, elevation, axis, sign in [(0,0,1,-1),(90,0,0,1),(0,90,0,0)]:
            r=model.control_snapshot(model.initial_state(),[0,0,0],azimuth,elevation)
            self.assertEqual((r['suggested_target'][axis]>0)-(r['suggested_target'][axis]<0),sign)
            for target in r['suggested_target']:
                self.assertAlmostEqual(target/.00001,round(target/.00001),places=8)

    def test_following_sampled_guidance_aligns_positive_z(self):
        for interval in [.1,1.0]:
            for azimuth,elevation in [(0,0),(90,0),(43,-35),(0,-90)]:
                state=model.initial_state()
                for sample in range(round(60/interval)):
                    target=model.control_snapshot(state,[0,0,0],azimuth,elevation)['suggested_target']
                    for _ in range(round(interval/.05)):
                        state,_=model.advance_targets(state,target,.05)
                result=model.snapshot(state,azimuth,elevation)
                self.assertLess(result['error_deg'],1.0,(interval,azimuth,elevation,result['error_deg']))
                self.assertLess(math.hypot(*state['omega']),.01)

    def test_random_tumble_can_be_detumbled_and_preserves_momentum(self):
        for seed in range(4):
            state=model.random_tumble(random.Random(seed))
            self.assertGreaterEqual(math.degrees(math.hypot(*state['omega'])),3)
            self.assertLessEqual(math.degrees(math.hypot(*state['omega'])),8.000001)
            self.assertEqual(state['h'],[0,0,0])
            original=model.snapshot(state)['momentum_world']
            for sample in range(25):
                targets=model.control_snapshot(state,[0,0,0],mode='detumble')['suggested_target']
                for _ in range(20):
                    state,_=model.advance_targets(state,targets,.05)
            self.assertLess(math.degrees(math.hypot(*state['omega'])),.2)
            for actual,expected in zip(model.snapshot(state)['momentum_world'],original):
                self.assertAlmostEqual(actual,expected,places=9)
            self.assertGreater(math.hypot(*state['h']),0)


if __name__ == '__main__':
    unittest.main()
