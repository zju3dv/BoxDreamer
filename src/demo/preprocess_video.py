import os
import subprocess
import glob
import tempfile
import shutil


def preprocess_videos(input_dir, output_dir=None, overwrite=False):
    """
    Convert videos in the specified directory to web browser-friendly formats

    Args:
        input_dir (str): Directory containing input video files
        output_dir (str, optional): Output directory, if not specified creates a 'processed' subdirectory
        overwrite (bool): Whether to overwrite existing processed files

    Returns:
        list: List of paths to processed video files
    """
    if output_dir is None:
        output_dir = os.path.join(input_dir, "processed")

    os.makedirs(output_dir, exist_ok=True)

    # Find all video files
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))

    processed_files = []

    for video_path in video_files:
        filename = os.path.basename(video_path)
        base_name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{base_name}_web.mp4")

        # Skip if already exists and not overwriting
        if os.path.exists(output_path) and not overwrite:
            print(f"File already exists, skipping: {output_path}")
            processed_files.append(output_path)
            continue

        print(f"Processing video: {video_path}")

        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        temp_output = os.path.join(temp_dir, f"{base_name}_temp.mp4")

        try:
            # Use ffmpeg to convert video
            # -c:v libx264: Use H.264 codec for video
            # -profile:v baseline: Use baseline profile for better compatibility
            # -level 3.0: Specify H.264 encoding level
            # -pix_fmt yuv420p: Use YUV 4:2:0 pixel format for better compatibility
            # -movflags +faststart: Optimize MP4 structure for smoother web playback
            # -c:a aac: Use AAC codec for audio
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                video_path,
                "-c:v",
                "libx264",
                "-profile:v",
                "baseline",
                "-level",
                "3.0",
                "-preset",
                "medium",
                "-tune",
                "fastdecode",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                "-vf",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # Ensure width/height are even
                "-movflags",
                "+faststart",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                temp_output,
            ]

            # Run ffmpeg command
            process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                print(f"Processing failed: {stderr.decode()}")
                continue

            # Move to target location after successful processing
            shutil.move(temp_output, output_path)
            processed_files.append(output_path)
            print(f"Processing completed: {output_path}")

        except Exception as e:
            print(f"Error processing video: {e}")

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    return processed_files


# Usage example
if __name__ == "__main__":
    # Process all videos in the specified directory
    video_dir = "src/demo/examples/mode1"
    processed_videos = preprocess_videos(video_dir)
    print(f"Processed videos: {processed_videos}")
