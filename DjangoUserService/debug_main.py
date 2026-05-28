import os
import sys


def main():
    """PyCharm debug entrypoint."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "DjangoUserService.settings")
    sys.argv = [sys.argv[0], "runserver", "127.0.0.1:8001", "--noreload"]

    from django.core.management import execute_from_command_line

    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
