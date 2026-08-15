import unittest

from safe_guard_ml import __version__


class PackageMetadataTest(unittest.TestCase):
    def test_version_is_public(self) -> None:
        self.assertEqual(__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
