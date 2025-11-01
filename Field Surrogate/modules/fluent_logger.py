#!/usr/bin/env python
"""
Fluent Logger - Separate Window Output Handler
==============================================
Manages Fluent output redirection to a separate console window.
"""

import sys
import subprocess
import tempfile
import os
from pathlib import Path
from threading import Thread
from datetime import datetime


class FluentLogger:
    """
    Handler for redirecting Fluent output to a separate console window.
    Captures stdout/stderr and redirects to separate window.
    """

    def __init__(self, title="Fluent Output"):
        """
        Initialize the Fluent logger.

        Parameters
        ----------
        title : str
            Title for the separate console window
        """
        self.title = title
        self.log_file = None
        self.process = None
        self.monitor_thread = None
        self._running = False
        self.original_stdout = None
        self.original_stderr = None

    def start(self):
        """Start the separate console window for Fluent output."""
        # Create temporary log file
        self.log_file = tempfile.NamedTemporaryFile(
            mode='w+',
            suffix='.log',
            prefix='fluent_',
            delete=False,
            buffering=1  # Line buffered
        )

        # Write header
        self.log_file.write("="*70 + "\n")
        self.log_file.write(f"{self.title}\n")
        self.log_file.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("="*70 + "\n\n")
        self.log_file.flush()

        # Launch separate console window to tail the log file
        if sys.platform == 'win32':
            # Windows: Use PowerShell with Get-Content -Wait (like tail -f)
            ps_cmd = f'Get-Content -Path "{self.log_file.name}" -Wait'
            self.process = subprocess.Popen(
                ['powershell', '-NoExit', '-Command', ps_cmd],
                creationflags=subprocess.CREATE_NEW_CONSOLE
            )
        else:
            # Linux/Mac: Use tail -f in new terminal
            self.process = subprocess.Popen(
                ['xterm', '-e', f'tail -f {self.log_file.name}']
            )

        self._running = True
        print(f"\n[Fluent Output] Redirected to separate window")
        print(f"  Log file: {self.log_file.name}")

    def write(self, message):
        """
        Write message to the log file (displayed in separate window).

        Parameters
        ----------
        message : str
            Message to write
        """
        if self.log_file and self._running:
            self.log_file.write(message)
            if not message.endswith('\n'):
                self.log_file.write('\n')
            self.log_file.flush()

    def stop(self):
        """Stop logging and close the separate window."""
        if self._running:
            self._running = False

            if self.log_file:
                self.log_file.write("\n" + "="*70 + "\n")
                self.log_file.write(f"Stopped: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.log_file.write("="*70 + "\n")
                self.log_file.close()

            # Don't kill the process - let user close it manually
            # This way they can review the log after completion
            print(f"\n[Fluent Output] Log window remains open for review")

    def get_log_path(self):
        """Get the path to the log file."""
        return self.log_file.name if self.log_file else None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def redirect_fluent_to_window(func):
    """
    Decorator to redirect Fluent-related print statements to separate window.

    Usage
    -----
    @redirect_fluent_to_window
    def run_simulation():
        # Fluent code here
        pass
    """
    def wrapper(*args, **kwargs):
        with FluentLogger("Fluent DOE Simulations") as logger:
            # Note: This is a simple implementation
            # For full redirection, would need to capture stdout/stderr
            return func(*args, **kwargs, fluent_logger=logger)
    return wrapper


if __name__ == "__main__":
    # Test the logger
    print("Testing Fluent Logger...")
    print("A separate console window should appear.")

    with FluentLogger("Test Fluent Output") as logger:
        logger.write("This is a test message")
        logger.write("Simulating Fluent output...")

        import time
        for i in range(10):
            logger.write(f"Iteration {i+1}/10")
            time.sleep(0.5)

        logger.write("Test complete!")

    print("\nTest finished. Check the separate window.")
    input("Press Enter to exit...")
