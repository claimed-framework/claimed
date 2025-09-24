import os


def test_cli():
    exit_status = os.system('terratorch iterate --help')
    assert exit_status == 0
