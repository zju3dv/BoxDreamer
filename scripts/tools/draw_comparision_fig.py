import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from PIL import Image
import numpy as np
import json


def save_selection(data_root, n, selected_scenes, selected_frameids):
    """Save current selection to backup file."""
    backup_data = {
        "data_root": data_root,
        "n": n,
        "selected_scenes": selected_scenes,
        "selected_frameids": selected_frameids,
    }

    # Save to hidden file in user's home directory
    backup_file = os.path.expanduser("~/.visual_grid_backup.json")
    try:
        with open(backup_file, "w") as f:
            json.dump(backup_data, f)
        print(f"Current selection saved to {backup_file}")
    except Exception as e:
        print(f"Unable to save backup: {e}")


def load_selection():
    """Load previous selection from backup file."""
    backup_file = os.path.expanduser("~/.visual_grid_backup.json")
    if not os.path.exists(backup_file):
        return None

    try:
        with open(backup_file, "r") as f:
            backup_data = json.load(f)
        print(f"Previous selection loaded")
        return backup_data
    except Exception as e:
        print(f"Unable to load backup: {e}")
        return None


def main():
    # Try to load previous selection
    previous_selection = load_selection()

    # Default values
    data_root = ""
    n = 0
    selected_scenes = []
    selected_frameids = []

    # If previous selection exists, ask user whether to use it
    if previous_selection:
        use_previous = input(
            f"Found previous selection (root directory: {previous_selection['data_root']}, columns: {previous_selection['n']})\nUse it? (y/n): "
        ).lower()

        if use_previous == "y" or use_previous == "yes":
            data_root = previous_selection["data_root"]
            n = previous_selection["n"]
            selected_scenes = previous_selection["selected_scenes"]
            selected_frameids = previous_selection["selected_frameids"]

            # Verify data root directory still exists
            if not os.path.isdir(data_root):
                print(f"Error: Previous root directory {data_root} no longer exists")
                data_root = ""  # Reset to empty to trigger re-selection below

    # If not using previous selection, get new input
    if not data_root:
        # Get data root directory
        data_root = input("Please enter data root directory path: ")

        # List available scene directories
        scenes = []
        try:
            scenes = [
                d
                for d in os.listdir(data_root)
                if os.path.isdir(os.path.join(data_root, d))
            ]
        except FileNotFoundError:
            print(f"Error: Directory {data_root} not found")
            return

        if not scenes:
            print(f"Error: No subdirectories found in {data_root}")
            return

        # Print scene directories and their indices
        print("Available scenes and their indices:")
        for i, scene in enumerate(scenes):
            print(f"{i}: {scene}")

        # Get user's desired number of columns n
        try:
            n = int(input("Please enter desired number of columns n: "))
            if n <= 0:
                print("Error: n must be a positive integer")
                return
        except ValueError:
            print("Error: Please enter a valid integer")
            return

        # Get user's selected scenes
        selected_scenes = []
        selected_frameids = []

        print(f"Please select {n} scenes in order (by index, repeats allowed):")
        for i in range(n):
            try:
                scene_idx = int(input(f"Enter scene index for column {i+1}: "))
                if scene_idx < 0 or scene_idx >= len(scenes):
                    print(f"Error: Index must be between 0 and {len(scenes)-1}")
                    return

                selected_scene = scenes[scene_idx]
                selected_scenes.append(selected_scene)

                # Check if the scene has a 'croped' directory
                croped_dir = os.path.join(data_root, selected_scene, "croped")
                if not os.path.isdir(croped_dir):
                    print(
                        f"Error: 'croped' directory not found in scene {selected_scene}"
                    )
                    return

                # List available frameids for the scene
                frameids = set()
                for file in os.listdir(croped_dir):
                    if file.endswith("-cropresults.png"):
                        frameid = file.split("-")[0]
                        frameids.add(frameid)

                if not frameids:
                    print(
                        f"Error: No valid image files found in 'croped' directory of scene {selected_scene}"
                    )
                    return

                print(
                    f"Available frameids for scene {selected_scene}: {', '.join(sorted(frameids))}"
                )
                frameid = input(f"Select a frameid for column {i+1}: ")

                if frameid not in frameids:
                    print(
                        f"Warning: frameid '{frameid}' may not exist in scene {selected_scene}"
                    )

                selected_frameids.append(frameid)

            except ValueError:
                print("Error: Please enter a valid integer index")
                return
    else:
        # Using previous selection, but need to verify scene list
        scenes = [
            d
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ]
        print("Using previous scene selection:")
        for i, scene in enumerate(selected_scenes):
            print(f"Column {i+1}: {scene}, frameid: {selected_frameids[i]}")

    # Find available method names
    all_methods = set()

    for scene_idx, scene in enumerate(selected_scenes):
        frameid = selected_frameids[scene_idx]
        croped_dir = os.path.join(data_root, scene, "croped")

        # Find files for specific frameid and extract method names
        for file in os.listdir(croped_dir):
            if file.startswith(f"{frameid}-") and file.endswith("-cropresults.png"):
                parts = file.split("-")
                if (
                    len(parts) >= 4
                ):  # Need at least frameid, color, methodname, cropresults
                    method = parts[-2]  # Method name is in the second-to-last position
                    all_methods.add(method)
                    print(f"Found method: {method} (file: {file})")

    # Sort other methods
    methods = []
    if "ours" in all_methods:
        all_methods.remove("ours")

    # Add other methods
    methods.extend(sorted(all_methods))

    # Add 'ours' at the end
    if "ours" in ["ours", "Ours"]:
        methods.append("ours")

    # Issue warning if not enough methods found
    if not methods:
        print("Error: No methods found")
        return

    if len(methods) < 3:
        print(
            f"Warning: Only found {len(methods)} methods ({', '.join(methods)}), fewer than expected 3"
        )

    # Use at most 3 methods
    methods = methods[: min(3, len(methods))]
    print(f"Will use the following methods: {', '.join(methods)}")

    # Save current selection for future use
    save_selection(data_root, n, selected_scenes, selected_frameids)

    # Create visualization
    visualize_grid(data_root, selected_scenes, selected_frameids, methods, n)


def visualize_grid(data_root, scenes, frameids, methods, n):
    # Configure academic paper style - enable LaTeX rendering
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times", "Times New Roman", "CMU Serif"],
                "font.size": 12,
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "axes.titlesize": 8,
                "axes.labelsize": 8,
            }
        )
        print("LaTeX rendering enabled")
    except:
        print("LaTeX rendering unavailable, using standard text rendering")
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": ["Times New Roman"],
                "font.size": 12,
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
                "axes.titlesize": 12,
                "axes.labelsize": 12,
            }
        )

    plt.style.use("seaborn-v0_8-whitegrid")

    # Create a rows×n figure
    rows = len(methods)

    # Use wider aspect ratio, focusing on compact display without considering paper width
    fig_width = n * 0.8  # Smaller width per column
    fig_height = rows * 0.8  # Height per row

    # Create figure
    fig = plt.figure(figsize=(fig_width, fig_height))

    # Use unified border color for each method
    method_colors = ["#7FB3B0", "#B3927F", "#A694C3"]

    # Method name width ratio - reserve less space for rotated method names
    method_name_width = 0.01  # Left method name width as proportion of total width

    # Settings to completely eliminate spacing
    left_margin = 0.1  # No left margin
    right_margin = 0  # No right margin
    top_margin = 0  # No top margin
    bottom_margin = 0  # No bottom margin

    # Calculate actual plotting area
    plot_width = 1.0 - left_margin - right_margin
    plot_height = 1.0 - top_margin - bottom_margin

    # Absolute no-spacing settings
    col_spacing = 0  # Absolutely no column spacing

    # Calculate width occupied by each image
    img_width = plot_width / n
    img_height = plot_height / rows

    # Method name mapping
    method_display_names = {
        "onepose": "OnePose++",
        "onepose++": "OnePose++",
        "gen6d": "Gen6D",
        "ours": "Ours",
    }

    # Add images to the figure
    for row, method in enumerate(methods):
        # Get correct method display name
        display_method = method_display_names.get(method.lower(), method)

        # Add method name on the left (using LaTeX format, rotated 90 degrees)
        # After rotation, text position needs adjustment
        x_pos = method_name_width / 3  # Horizontal position aligned to left
        y_pos = 1.0 - top_margin - (row + 0.5) * img_height  # Vertically centered

        # Format method name using LaTeX
        if plt.rcParams["text.usetex"]:
            method_text = r"\textrm{" + display_method + "}"
        else:
            method_text = display_method

        # Add rotated method name (counter-clockwise 90 degrees)
        plt.figtext(
            x_pos,
            y_pos,
            method_text,
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            fontweight="normal",
            rotation=90,
        )

        for col in range(n):
            scene = scenes[col]
            frameid = frameids[col]

            # Calculate subplot position - no spacing
            x_left = method_name_width + col * img_width
            y_bottom = 1.0 - top_margin - (row + 1) * img_height
            width = img_width
            height = img_height

            # Create subplot
            ax = fig.add_axes([x_left, y_bottom, width, height])

            # Find corresponding image file
            croped_dir = os.path.join(data_root, scene, "croped")
            matching_file = None

            for file in os.listdir(croped_dir):
                if file.startswith(f"{frameid}-") and file.endswith("-cropresults.png"):
                    parts = file.split("-")
                    if len(parts) >= 4 and parts[-2] == method:
                        matching_file = file
                        break

            if matching_file:
                # Use the matching file found
                img_path = os.path.join(croped_dir, matching_file)
                try:
                    img = Image.open(img_path)
                    ax.imshow(np.array(img))
                    # Very thin border
                    for spine in ax.spines.values():
                        spine.set_color(method_colors[row % len(method_colors)])
                        spine.set_linewidth(0.8)  # Use a very thin border
                except Exception as e:
                    print(f"Unable to load image {img_path}: {e}")
                    ax.text(0.5, 0.5, "Error", ha="center", va="center", fontsize=6)
            else:
                # If no matching file found
                ax.text(0.5, 0.5, "No image", ha="center", va="center", fontsize=6)

            # Remove axes and grid
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)

            # Eliminate all internal padding
            ax.set_xmargin(0)
            ax.set_ymargin(0)

    # Save as PDF and PNG, using compact layout
    output_path_pdf = os.path.join(data_root, "visualization_grid.pdf")
    plt.savefig(
        output_path_pdf, dpi=300, bbox_inches="tight", pad_inches=0, format="pdf"
    )
    print(f"Visualization saved to: {output_path_pdf}")

    output_path_png = os.path.join(data_root, "visualization_grid.png")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Visualization also saved as PNG: {output_path_png}")

    # Display image
    plt.show()


if __name__ == "__main__":
    main()
