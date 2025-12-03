#!/usr/bin/env python3
"""
Collect all plot files (png, pdf, eps) from dicke directory into dicke/plots subdirectory.
Simple, direct file organization script.
"""

import os
import shutil
from pathlib import Path

def collect_plots():
    """Move all plot files to plots subdirectory."""
    # Define plot extensions
    plot_extensions = {'.png', '.pdf', '.eps'}
    
    # Get current directory (dicke)
    current_dir = Path('.')
    
    # Create plots directory if it doesn't exist
    plots_dir = current_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Find and move plot files
    moved_count = 0
    for file_path in current_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in plot_extensions:
            destination = plots_dir / file_path.name
            
            # Handle name conflicts by adding counter
            counter = 1
            original_dest = destination
            while destination.exists():
                stem = original_dest.stem
                suffix = original_dest.suffix
                destination = plots_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Move the file
            shutil.move(str(file_path), str(destination))
            print(f"Moved: {file_path.name} -> plots/{destination.name}")
            moved_count += 1
    
    print(f"\nMoved {moved_count} plot files to plots/ directory.")

if __name__ == "__main__":
    collect_plots()
