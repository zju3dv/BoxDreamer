#!/usr/bin/env python3
import os
import argparse
import cv2
import glob
from tqdm import tqdm
import numpy as np

def create_video_from_images(image_folder, output_file, fps, extensions=None):
    """
    Create a video from a sequence of images in a folder.
    
    Parameters:
    image_folder (str): Path to the folder containing images
    output_file (str): Path to the output video file
    fps (int): Frames per second for the output video
    extensions (list): List of image file extensions to include
    """
    if extensions is None:
        extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
    
    # Create a list of all image files
    image_files = []
    for ext in extensions:
        pattern = os.path.join(image_folder, f'*.{ext}')
        image_files.extend(glob.glob(pattern))
        # Also check for uppercase extensions
        pattern = os.path.join(image_folder, f'*.{ext.upper()}')
        image_files.extend(glob.glob(pattern))
    
    if not image_files:
        print(f"No image files found in {image_folder} with extensions: {extensions}")
        return False
    
    # Sort files by name
    image_files.sort()
    
    print(f"Found {len(image_files)} images")
    
    # Read the first image to get dimensions
    first_image = cv2.imread(image_files[0])
    if first_image is None:
        print(f"Could not read image: {image_files[0]}")
        return False
    
    height, width, layers = first_image.shape
    size = (width, height)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use 'XVID' for .avi
    out = cv2.VideoWriter(output_file, fourcc, fps, size)
    
    # Write images to video
    print("Creating video...")
    for image_file in tqdm(image_files):
        img = cv2.imread(image_file)
        if img is not None:
            # Check if dimensions match
            if img.shape[0:2] != (height, width):
                print(f"Warning: Image {image_file} has different dimensions. Resizing...")
                img = cv2.resize(img, size)
            out.write(img)
        else:
            print(f"Warning: Could not read image {image_file}")
    
    out.release()
    print(f"Video saved as {output_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Create a video from a sequence of images.')
    parser.add_argument('folder', type=str, help='Folder containing image files')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output video file (default: video.mp4 in the image folder)')
    parser.add_argument('--fps', '-f', type=int, default=30, 
                        help='Frames per second (default: 30)')
    parser.add_argument('--extensions', '-e', type=str, default=None,
                        help='Comma-separated list of image extensions to include')
    
    args = parser.parse_args()
    
    # Check if the folder exists
    if not os.path.isdir(args.folder):
        print(f"Error: The folder '{args.folder}' does not exist.")
        return
    
    # Set default output file if not specified
    if args.output is None:
        args.output = os.path.join(args.folder, 'video.mp4')
    
    # Parse extensions
    extensions = None
    if args.extensions:
        extensions = [ext.strip() for ext in args.extensions.split(',')]
    
    # Create video
    success = create_video_from_images(args.folder, args.output, args.fps, extensions)
    
    if success:
        print("Video creation completed successfully!")
    else:
        print("Video creation failed.")

if __name__ == "__main__":
    main()