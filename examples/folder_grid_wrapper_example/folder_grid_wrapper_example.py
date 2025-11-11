from pathlib import Path

def grid_process(source_folder: str, target_folder: str) -> None:
    src_dir = Path(source_folder)
    tgt_dir = Path(target_folder)

    for src_file in sorted(src_dir.glob("*.txt")):
        text = src_file.read_text(encoding="utf-8")
        updated = text.replace("test", "test processed")
        (tgt_dir / src_file.name).write_text(updated, encoding="utf-8")