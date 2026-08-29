from pathlib import Path

import pytest

from motion_capture.config import (
    build_ndi_config,
    build_realsense_config,
    load_config,
    save_ndi_config,
    save_realsense_config,
    save_source_mode,
)


def test_loads_relative_paths_from_project_root(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "MOCAP_SOURCE=ndi\n"
        "APRILTAG_IDS=10,11\n"
        "APRILTAG_BOARD_ROWS=1\n"
        "APRILTAG_BOARD_COLS=2\n"
        "NDI_TRACKER_TYPE=vega\n"
        "NDI_ROM_FILES=roms/a.rom,roms/b.rom\n",
        encoding="utf-8",
    )
    config = load_config(tmp_path, environ={})
    assert config.source == "ndi"
    assert config.realsense.tag_ids == (10, 11)
    assert config.ndi.rom_files == (tmp_path / "roms/a.rom", tmp_path / "roms/b.rom")
    assert config.open3d.enabled is True
    assert config.open3d.render_hz == 15


def test_rejects_board_shape_mismatch(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "APRILTAG_IDS=0,1,2\nAPRILTAG_BOARD_ROWS=2\nAPRILTAG_BOARD_COLS=2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="数量必须等于"):
        load_config(tmp_path, environ={})


def test_rejects_invalid_open3d_boolean(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPEN3D_ENABLED=maybe\n", encoding="utf-8")
    with pytest.raises(ValueError, match="OPEN3D_ENABLED"):
        load_config(tmp_path, environ={})


def test_saves_ndi_form_settings_to_root_env(tmp_path: Path) -> None:
    ndi = build_ndi_config(
        root=tmp_path,
        tracker_type="VEGA",
        ip_address="192.168.10.20",
        port=8765,
        serial_port="",
        rom_files=("roms/tool_a.rom", "roms/tool_b.ROM"),
        record_orientation="matrix",
        record_format="xlsx",
    )

    env_path = save_ndi_config(tmp_path, ndi)
    loaded = load_config(tmp_path, environ={})

    assert env_path == tmp_path / ".env"
    assert loaded.ndi.ip_address == "192.168.10.20"
    assert loaded.ndi.rom_files == (
        tmp_path / "roms/tool_a.rom",
        tmp_path / "roms/tool_b.ROM",
    )
    assert loaded.ndi.record_orientation == "matrix"
    assert loaded.ndi.record_format == "xlsx"
    assert "NDI_ROM_FILES='roms/tool_a.rom,roms/tool_b.ROM'" in env_path.read_text(encoding="utf-8")


def test_ndi_form_rejects_invalid_vega_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="IP 地址"):
        build_ndi_config(tmp_path, "vega", "", 8765, "", ("tool.rom",))
    with pytest.raises(ValueError, match=".rom"):
        build_ndi_config(tmp_path, "vega", "127.0.0.1", 8765, "", ("tool.txt",))


def test_saves_realsense_form_and_reference_to_root_env(tmp_path: Path) -> None:
    reference = (
        1.0, 0.0, 0.0, 120.0,
        0.0, 1.0, 0.0, -35.0,
        0.0, 0.0, 1.0, 650.0,
        0.0, 0.0, 0.0, 1.0,
    )
    realsense = build_realsense_config(
        serial="D435I-TEST",
        width=1280,
        height=720,
        fps=30,
        tag_family="tag36h11",
        tag_ids=(0, 1, 2, 3),
        board_rows=2,
        board_cols=2,
        tag_size_m=0.08,
        tag_spacing_m=0.02,
        min_visible_tags=2,
        record_format="xlsx",
        reference_transform=reference,
    )

    save_realsense_config(tmp_path, realsense)
    loaded = load_config(tmp_path, environ={})

    assert loaded.realsense.serial == "D435I-TEST"
    assert loaded.realsense.reference_transform == reference
    assert loaded.realsense.calibration_samples == 30
    assert loaded.realsense.min_visible_tags == 2
    assert loaded.realsense.record_format == "xlsx"


def test_realsense_rejects_minimum_visible_tags_above_board_size() -> None:
    with pytest.raises(ValueError, match="MIN_VISIBLE_TAGS"):
        build_realsense_config(
            serial="",
            width=1280,
            height=720,
            fps=30,
            tag_family="tag36h11",
            tag_ids=(0, 1, 2, 3),
            board_rows=2,
            board_cols=2,
            tag_size_m=0.08,
            tag_spacing_m=0.02,
            min_visible_tags=5,
        )


def test_saves_selected_work_mode(tmp_path: Path) -> None:
    save_source_mode(tmp_path, "RealSense")
    assert load_config(tmp_path, environ={}).source == "realsense"
    with pytest.raises(ValueError, match="工作模式"):
        save_source_mode(tmp_path, "camera")
