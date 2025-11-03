#!/usr/bin/env python
"""
Fluent Cleanup Utilities
========================
Handles cleanup operations for Fluent sessions and temporary files.
"""

from pathlib import Path


def end_fluent_session(solver_session, verbose=True):
    """
    Safely close a Fluent solver session.

    Parameters
    ----------
    solver_session : Fluent solver object
        The active Fluent session to close
    verbose : bool
        Print status messages
    """
    if verbose:
        print(f"\n[Cleanup] Closing Fluent session...")

    try:
        solver_session.exit()
        if verbose:
            print(f"  ✓ Fluent session closed successfully")
    except Exception as e:
        if verbose:
            print(f"  Warning: Error closing Fluent: {e}")


def cleanup_trn_files(directory, verbose=True):
    """
    Remove all .trn files from specified directory.

    Parameters
    ----------
    directory : Path or str
        Directory to clean
    verbose : bool
        Print status messages

    Returns
    -------
    int
        Number of files deleted
    """
    directory = Path(directory)

    if verbose:
        print(f"\n[Cleanup] Removing Fluent .trn files from {directory.name}/...")

    trn_files = list(directory.glob("*.trn"))
    deleted_count = 0

    if trn_files:
        for trn_file in trn_files:
            try:
                trn_file.unlink()
                if verbose:
                    print(f"  Deleted: {trn_file.name}")
                deleted_count += 1
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not delete {trn_file.name}: {e}")

        if verbose:
            print(f"  ✓ Removed {deleted_count} .trn file(s)")
    else:
        if verbose:
            print(f"  No .trn files found")

    return deleted_count


def cleanup_pycache(directory, verbose=True):
    """
    Remove __pycache__ directory.

    Parameters
    ----------
    directory : Path or str
        Directory to clean
    verbose : bool
        Print status messages

    Returns
    -------
    bool
        True if deleted, False otherwise
    """
    directory = Path(directory)
    pycache = directory / "__pycache__"

    if pycache.exists():
        if verbose:
            print(f"\n[Cleanup] Removing Python cache...")

        try:
            import shutil
            shutil.rmtree(pycache)
            if verbose:
                print(f"  ✓ Removed __pycache__/")
            return True
        except Exception as e:
            if verbose:
                print(f"  Warning: Could not remove __pycache__: {e}")
            return False

    return False


def full_cleanup(directory, close_fluent=None, verbose=True):
    """
    Perform complete cleanup of Fluent session and temporary files.

    Parameters
    ----------
    directory : Path or str
        Project directory to clean
    close_fluent : Fluent solver object, optional
        If provided, will close this Fluent session
    verbose : bool
        Print status messages
    """
    if close_fluent is not None:
        end_fluent_session(close_fluent, verbose=verbose)

    cleanup_trn_files(directory, verbose=verbose)
    cleanup_pycache(directory, verbose=verbose)

    if verbose:
        print(f"\n✓ Cleanup complete")


if __name__ == "__main__":
    # Standalone cleanup mode
    import sys

    print("="*70)
    print("FLUENT CLEANUP UTILITY")
    print("="*70)

    project_dir = Path(__file__).parent
    print(f"\nProject directory: {project_dir}")

    # Check for files to clean
    trn_count = len(list(project_dir.glob("*.trn")))
    pycache_exists = (project_dir / "__pycache__").exists()

    print(f"\nFiles to clean:")
    print(f"  .trn files: {trn_count}")
    print(f"  __pycache__: {'Yes' if pycache_exists else 'No'}")

    if trn_count == 0 and not pycache_exists:
        print("\nNothing to clean!")
        sys.exit(0)

    response = input("\nProceed with cleanup? [y/N]: ").strip().lower()
    if response == 'y':
        cleanup_trn_files(project_dir, verbose=True)
        cleanup_pycache(project_dir, verbose=True)
        print("\n✓ Cleanup completed!")
    else:
        print("\nCleanup cancelled.")
