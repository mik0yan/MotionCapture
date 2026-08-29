from pathlib import Path

from motion_capture.rom_importer import import_rom_files


def test_imports_roms_and_reuses_identical_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    source = source_dir / "tool.rom"
    source.write_bytes(b"rom-a")

    first = import_rom_files(tmp_path, (source,))
    second = import_rom_files(tmp_path, (source,))

    assert first.copied_count == 1
    assert first.paths == (tmp_path / "roms/tool.rom",)
    assert second.reused_count == 1
    assert second.paths[0].read_bytes() == b"rom-a"


def test_import_renames_conflicting_rom_without_overwrite(tmp_path: Path) -> None:
    existing_dir = tmp_path / "roms"
    existing_dir.mkdir()
    (existing_dir / "tool.rom").write_bytes(b"existing")
    incoming = tmp_path / "tool.rom"
    incoming.write_bytes(b"new")

    result = import_rom_files(tmp_path, (incoming,))

    assert (existing_dir / "tool.rom").read_bytes() == b"existing"
    assert result.paths == (existing_dir / "tool_2.rom",)
    assert result.renamed_count == 1
    assert result.paths[0].read_bytes() == b"new"
