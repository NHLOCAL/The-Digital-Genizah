import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from upload_directory_to_hf import upload_directory


class FakeHfApi:
    def __init__(self):
        self.calls = []

    def delete_files(self, **kwargs):
        self.calls.append(("delete_files", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))


class UploadDirectoryTests(unittest.TestCase):
    def test_upload_directory_replaces_shards_in_one_upload_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "part-00001.parquet").write_bytes(b"parquet fixture")
            api = FakeHfApi()

            upload_directory(
                repo_id="NHLOCAL/judaic-texts-corpus",
                local_dir=Path(tmp),
                path_in_repo="data",
                commit_message="Update Otzaria dataset",
                delete_patterns=["data/*.parquet"],
                api=api,
                token="token",
            )

        self.assertEqual([name for name, _ in api.calls], ["upload_folder"])
        upload_kwargs = api.calls[0][1]
        self.assertEqual(upload_kwargs["repo_id"], "NHLOCAL/judaic-texts-corpus")
        self.assertEqual(upload_kwargs["repo_type"], "dataset")
        self.assertEqual(upload_kwargs["path_in_repo"], "data")
        self.assertEqual(upload_kwargs["commit_message"], "Update Otzaria dataset")
        self.assertEqual(upload_kwargs["delete_patterns"], ["*.parquet"])

    def test_upload_directory_skips_delete_call_without_patterns(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeHfApi()

            upload_directory(
                repo_id="nhlocal/otzar-hatorah",
                local_dir=Path(tmp),
                path_in_repo="data",
                commit_message="Update",
                delete_patterns=[],
                api=api,
                token="token",
            )

        self.assertEqual([name for name, _ in api.calls], ["upload_folder"])

    def test_refuses_empty_local_parquet_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeHfApi()
            with self.assertRaisesRegex(ValueError, "No local Parquet"):
                upload_directory(
                    repo_id="NHLOCAL/judaic-texts-corpus",
                    local_dir=Path(tmp),
                    path_in_repo="data",
                    delete_patterns=["data/*.parquet"],
                    api=api,
                    token="token",
                )
            self.assertEqual(api.calls, [])

    def test_refuses_delete_patterns_outside_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "part.parquet").write_bytes(b"data")
            api = FakeHfApi()
            with self.assertRaisesRegex(ValueError, "outside destination"):
                upload_directory(
                    repo_id="NHLOCAL/judaic-texts-corpus",
                    local_dir=Path(tmp),
                    path_in_repo="data",
                    delete_patterns=["other/*.parquet"],
                    api=api,
                    token="token",
                )
            self.assertEqual(api.calls, [])

    def test_upload_directory_requires_token(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "HUGGINGFACE_TOKEN"):
                upload_directory(
                    repo_id="NHLOCAL/judaic-texts-corpus",
                    local_dir=Path(tmp),
                    path_in_repo="data",
                    commit_message="Update",
                    delete_patterns=[],
                )


if __name__ == "__main__":
    unittest.main()
