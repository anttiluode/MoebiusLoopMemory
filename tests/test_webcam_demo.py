"""Headless webcam tests. Install requirements-webcam.txt first."""
import unittest
import numpy as np
import cv2
from webcam_demo import FrameEncoder, LiveMemory, make_canvas, simulated_frames


class WebcamTests(unittest.TestCase):
    def test_unchanged_frame_is_zero_input_in_change_mode(self):
        enc=FrameEncoder()
        image=np.zeros((480,640,3),np.uint8)
        cv2.rectangle(image,(100,120),(220,240),(250,250,250),-1)
        self.assertTrue(np.allclose(enc.encode(image,'change'),0))
        moved=image.copy()
        cv2.rectangle(moved,(220,120),(300,240),(250,250,250),-1)
        self.assertGreater(np.linalg.norm(enc.encode(moved,'change')),.1)
        self.assertTrue(np.allclose(enc.encode(moved,'change'),0))

    def test_webcam_state_uses_exact_same_core_update(self):
        memory=LiveMemory()
        signal=np.linspace(-.7,.7,16)
        memory.update(signal)
        expected=memory.gated_net.advance(np.zeros(16),signal)
        self.assertTrue(np.allclose(memory.gated,expected))
        self.assertTrue(np.allclose(memory.ungated,
            memory.ungated_net.advance(np.zeros(16),signal)))
        memory.update(signal)
        self.assertTrue(np.allclose(memory.gated,memory.gated_net.advance(expected,signal)))

    def test_hold_zero_input_clears_state_over_time(self):
        mem=LiveMemory()
        for _ in range(10):
            mem.update(np.full(16,.1))
        initial=np.linalg.norm(mem.gated)
        for _ in range(100):
            mem.update(np.zeros(16))
        self.assertLess(np.linalg.norm(mem.gated),initial*.001)

    def test_static_appearance_keeps_driving(self):
        enc=FrameEncoder()
        img=np.zeros((480,640,3),np.uint8)
        img[:,:320]=255
        a=enc.encode(img,'appearance')
        b=enc.encode(img,'appearance')
        self.assertGreater(np.linalg.norm(a),.1)
        self.assertTrue(np.allclose(a,b))

    def test_control_shares_geometry_and_projection(self):
        mem=LiveMemory()
        self.assertTrue(np.allclose(mem.gated_net.R,mem.ungated_net.R))
        self.assertTrue(np.allclose(mem.gated_net.B,mem.ungated_net.B))
        mem.update(np.linspace(-.5,.5,16))
        before=mem.gated.copy()
        _=mem.probe_response(mem.gated_net,mem.gated,mem.fixed_probe)
        self.assertTrue(np.allclose(before,mem.gated))
        self.assertTrue(np.isfinite(mem.gated_probe))

    def test_render_headless(self):
        enc=FrameEncoder()
        mem=LiveMemory()
        img=next(simulated_frames())
        mem.update(enc.encode(img))
        canvas=make_canvas(img,enc,mem,'change',False,False,15,15)
        self.assertEqual(canvas.shape,(760,1280,3))
        self.assertGreater(int(canvas.max()),100)

    def test_reset(self):
        mem=LiveMemory()
        mem.update(np.ones(16))
        mem.reset()
        self.assertEqual(mem.step,0)
        self.assertEqual(len(mem.gated_history),0)
        self.assertTrue(np.allclose(mem.gated,0))


if __name__ == '__main__':
    unittest.main()
