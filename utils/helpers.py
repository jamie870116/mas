import cv2
import os
from pathlib import Path
import re

def save_to_video(file_name: str, fps: int = 10, project_root: str = None):
    """
    Convert saved frames into videos for each agent's POV and overhead view.
    
    Args:
        file_name (str): The task folder name (e.g., 'task1/test_4') or path to the task folder.
            If relative and project_root is not provided, it resolves relative to the directory of this script.
            If absolute, it uses the provided path directly.
        fps (int): Frames per second for the output video (default: 30).
        project_root (str, optional): The root directory of the project (e.g., '/Users/apple/Desktop/UCSB/master_project/mas').
            If provided, file_name is resolved relative to this path.
    """
    # Define the task folder path
    if project_root:
        task_path = Path(project_root) / file_name
    else:
        # Resolve relative to the directory of this script (utils), then go up to mas
        script_dir = Path(__file__).parent  # /mas/utils
        task_path = (script_dir / ".." / file_name).resolve()  # Go up to /mas, then to task1/test_4
    
    print(f"Resolved task path: {task_path}")
    
    if not task_path.exists() or not task_path.is_dir():
        raise ValueError(f"Task folder {task_path} does not exist or is not a directory.")
    
    # Find all subfolders (e.g., Alice/pov, Bob/pov, overhead)
    subfolders = []
    # Look for agent POV folders (e.g., Alice/pov)
    for agent_folder in task_path.iterdir():
        if agent_folder.is_dir() and (agent_folder / "pov").exists():
            subfolders.append(agent_folder / "pov")
        elif agent_folder.name == "overhead" and agent_folder.is_dir():
            subfolders.append(agent_folder)
    
    if not subfolders:
        raise ValueError(f"No valid subfolders (agent POV or overhead) found in {task_path}.")
    
    # Process each subfolder to create a video
    for subfolder in subfolders:
        # Collect all frame files in the subfolder
        frame_files = sorted(
            [f for f in subfolder.iterdir() if f.is_file() and f.suffix == ".png"],
            key=lambda x: int(re.search(r'frame_(\d+)\.png', x.name).group(1))
        )
        
        if not frame_files:
            print(f"No frames found in {subfolder}. Skipping video creation.")
            continue
        
        # Read the first frame to get dimensions
        first_frame = cv2.imread(str(frame_files[0]))
        height, width, _ = first_frame.shape
        
        # Define the output video path
        # E.g., if subfolder is task1/test_4/Alice/pov, video will be task1/test_4/Alice_pov.mp4
        video_name = subfolder.name if subfolder.name == "overhead" else f"{subfolder.parent.name}_{subfolder.name}"
        video_path = task_path / f"{video_name}.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
        video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        # Write each frame to the video
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            video_writer.write(frame)
        
        # Release the video writer
        video_writer.release()
        print(f"Video saved at {video_path}")

if __name__ == "__main__":
    # Example usage with relative path
    save_to_video("logs/bring_a_tomato,_lettuce,_and_bread_to_the_countertop_to_make_a_sandwich/test_16")