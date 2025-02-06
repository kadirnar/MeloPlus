import shutil
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


class AudioProcessor:

    def __init__(self, parquet_path: Union[str, Path]) -> None:
        self.parquet_path = Path(parquet_path)
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            try:
                if self.parquet_path.is_file():
                    print(f"Loading single parquet file: {self.parquet_path}")
                    self._df = pd.read_parquet(self.parquet_path)
                else:
                    print(f"Loading parquet files from directory: {self.parquet_path}")
                    parquet_files = list(self.parquet_path.glob("*.parquet"))
                    if not parquet_files:
                        raise ValueError(f"No parquet files found in {self.parquet_path}")

                    dfs = []
                    for file in parquet_files:
                        print(f"Loading parquet file: {file}")
                        dfs.append(pd.read_parquet(file))
                    self._df = pd.concat(dfs, ignore_index=True)

            except Exception as err:
                raise ValueError(f"Failed to read parquet file(s): {err}")
        return self._df

    def extract_audio_files(
            self,
            output_dir: Union[str, Path],
            columns_to_extract: Optional[list[str]] = None,
            limit: Optional[int] = None) -> list[Path]:
        try:
            output_dir = Path(output_dir)
            audio_dir = output_dir / "wavs"
            audio_dir.mkdir(parents=True, exist_ok=True)

            df = self.df
            available_columns = set(df.columns)
            print(f"Available columns in dataset: {available_columns}")

            if columns_to_extract is None:
                columns_to_extract = [
                    col for col in available_columns if col != 'audio' and not col.startswith('__')
                ]
            else:
                invalid_columns = set(columns_to_extract) - available_columns
                if invalid_columns:
                    raise ValueError(f"Column(s) not found in dataset: {invalid_columns}")

            print(f"Columns to extract: {columns_to_extract}")

            column_dirs = {}
            for col in columns_to_extract:
                col_dir = output_dir / col
                col_dir.mkdir(parents=True, exist_ok=True)
                column_dirs[col] = col_dir

            if limit is not None:
                df = df.head(limit)

            total_files = len(df)
            progress = tqdm(
                df.iterrows(),
                total=total_files,
                desc="Extracting files",
            )

            audio_files: list[Path] = []
            for idx, row in progress:
                audio_data = row.get('audio', {})
                if isinstance(audio_data, dict):
                    audio_bytes = audio_data.get('bytes', b'')
                else:
                    audio_bytes = audio_data

                filename = str(row.get('id', row.get('filename', len(audio_files))))

                audio_path = audio_dir / f"{filename}.wav"
                if isinstance(audio_bytes, bytes):
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    audio_files.append(audio_path)

                    for col, col_dir in column_dirs.items():
                        content = row.get(col, '')
                        if content:
                            file_path = col_dir / f"{filename}.txt"
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(str(content))

            print(f"Extracted {len(audio_files)} files to:")
            print(f"  - Audio files: {audio_dir}")
            for col, col_dir in column_dirs.items():
                print(f"  - {col} files: {col_dir}")
            return audio_files

        except Exception as err:
            raise ValueError(f"Failed to extract files: {err}")

    def get_metadata(self) -> dict[str, Union[int, list[str], pd.DataFrame]]:
        try:
            df = self.df
            total_size = df.memory_usage(deep=True).sum()
            metadata = {
                "total_files": len(df),
                "file_size_mb": total_size / (1024 * 1024),
                "columns": list(df.columns),
                "sample": df.head(5),
            }
            return metadata
        except Exception as err:
            raise ValueError(f"Failed to get metadata: {err}")


class HFDatasetManager:

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token
        self.api = HfApi(token=token)

    def download(
        self,
        repo_id: str,
        local_dir: Union[str, Path],
        filename: Optional[str] = None,
        repo_type: str = "dataset",
        ignore_patterns: Optional[list[str]] = None,
        no_cache: bool = False,
    ) -> Path:
        try:
            local_dir = Path(local_dir)
            local_dir.mkdir(parents=True, exist_ok=True)

            default_ignore = [".git*", "README.md", "*.md", "LICENSE", "__pycache__", "*.pyc", ".DS_Store"]

            if ignore_patterns:
                default_ignore.extend(ignore_patterns)

            if filename:
                file_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type=repo_type,
                    token=self.token,
                    local_dir=local_dir,
                    local_dir_use_symlinks=False,
                )
                print(f"Downloaded file: {file_path}")
                return Path(file_path)
            else:
                dataset_path = snapshot_download(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    token=self.token,
                    local_dir=local_dir,
                    ignore_patterns=default_ignore,
                    local_dir_use_symlinks=False,
                )

                cache_dir = Path(local_dir) / ".cache"
                if cache_dir.exists():
                    print(f"Removing cache directory: {cache_dir}")
                    shutil.rmtree(cache_dir)

                print(f"Downloaded dataset to: {dataset_path}")
                return Path(dataset_path)

        except Exception as err:
            raise RuntimeError(f"Failed to download dataset: {err}")

    def upload(
        self,
        local_path: Union[str, Path],
        repo_id: str,
        repo_type: str = "dataset",
    ) -> None:
        try:
            local_path = Path(local_path)
            print(f"Uploading {local_path} to {repo_id}")

            if local_path.is_file():
                self.api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=local_path.name,
                    repo_id=repo_id,
                    repo_type=repo_type,
                )
            elif local_path.is_dir():
                self.api.upload_folder(
                    folder_path=str(local_path),
                    repo_id=repo_id,
                    repo_type=repo_type,
                )

            print("Upload complete")
        except Exception as e:
            raise RuntimeError(f"Failed to upload to {repo_id}: {str(e)}")


def process_audio_data(
    dataset_name: str = "bookbot/ljspeech_phonemes",
    output_dir: str = "output/ljspeech_phonemes",
    columns_to_extract: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> None:
    try:
        dataset = HFDatasetManager()
        dataset.download(
            repo_id=dataset_name,
            local_dir=output_dir,
            repo_type="dataset",
        )

        data_dir = Path(output_dir) / "data"
        processor = AudioProcessor(data_dir)

        metadata = processor.get_metadata()
        print(f"Dataset metadata: {metadata}")

        df = processor.df
        print(f"Column names: {df.columns}")
        print(f"First row: {df.head(1)}")

        audio_files = processor.extract_audio_files(
            output_dir=str(Path(output_dir) / "audio_files"),
            columns_to_extract=columns_to_extract,
            limit=limit,
        )
        print(f"Extracted {len(audio_files)} audio files")

    except Exception as err:
        raise RuntimeError(f"Failed to process audio data: {err}")


if __name__ == "__main__":
    process_audio_data(
        dataset_name="bookbot/ljspeech_phonemes",
        output_dir="output/ljspeech_phonemes",
        columns_to_extract=["text", "phonemes"],
        limit=None,
    )
