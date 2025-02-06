from pathlib import Path
import shutil


def create_metadata_list(
        source_audio_dir: str,
        text_dir: str,
        output_dir: str,
        language: str = "EN",
        speaker: str = "default") -> None:
    """
    Create metadata list for MeloPlus from audio and text files.

    Args:
        source_audio_dir (str): Source directory containing wav files
        text_dir (str): Directory containing text files
        output_dir (str): Output directory for metadata and copied wav files
        language (str, optional): Language code. Defaults to "EN".
        speaker (str, optional): Speaker identifier. Defaults to "default".
    """
    # Convert paths to Path objects
    source_audio_dir = Path(source_audio_dir)
    text_dir = Path(text_dir)
    output_dir = Path(output_dir)

    # Define target directories
    target_audio_dir = output_dir / "wavs"
    output_file = output_dir / "metadata.list"

    # Create necessary directories
    target_audio_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Create metadata entries
    metadata_lines = []

    # Get all wav files and sort them
    wav_files = sorted(list(source_audio_dir.glob("*.wav")))

    print(f"Processing {len(wav_files)} audio files...")

    for wav_path in wav_files:
        # Get corresponding text file
        text_file = text_dir / f"{wav_path.stem}.txt"

        if text_file.exists():
            # Read the text content
            with open(text_file, encoding="utf-8") as f:
                text = f.read().strip()

            # Copy wav file to target directory
            target_wav_path = target_audio_dir / wav_path.name
            shutil.copy2(wav_path, target_wav_path)

            # Create metadata line with the correct relative path
            relative_wav_path = f"wavs/{wav_path.name}"
            metadata_line = f"{relative_wav_path}|{language}-{speaker}|{language}|{text}"
            metadata_lines.append(metadata_line)

    # Write metadata file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(metadata_lines))

    print(f"Created metadata file at: {output_file}")
    print(f"Copied audio files to: {target_audio_dir}")
    print(f"Total entries: {len(metadata_lines)}")


if __name__ == "__main__":
    create_metadata_list(
        source_audio_dir="output/ljspeech_phonemes/audio_files/wavs",
        text_dir="output/ljspeech_phonemes/audio_files/text",
        output_dir="meloplus/data/example")
