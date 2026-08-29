from motion_capture.backends.simulator import SimulatorBackend
from motion_capture.config import load_config


def test_simulator_produces_independent_tag_samples_and_detections(tmp_path) -> None:
    config = load_config(tmp_path, environ={})
    backend = SimulatorBackend(config.realsense)
    backend.start()
    packet = backend.read()
    backend.stop()

    assert packet.image_rgb is not None
    assert packet.image_rgb.shape == (720, 1280, 3)
    assert packet.tag_detections
    assert {sample.tool_id for sample in packet.samples} >= {"tag_00", "apriltag_board"}
    assert packet.tag_detections[0].tag_id == 0
    assert len(packet.tag_detections[0].corners_px) == 4
