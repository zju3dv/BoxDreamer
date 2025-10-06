"""
BoxDreamer CLI - Command-line interface for BoxDreamer demo
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.demo.demo import main as demo_main


def main():
    """CLI entry point for boxdreamer-cli command"""
    try:
        demo_main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
