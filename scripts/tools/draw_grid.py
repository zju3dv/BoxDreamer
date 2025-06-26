import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
from PIL import Image
import numpy as np
import json


def save_selection(image_paths, image_descs):
    """Save current selection to backup file."""
    backup_data = {"image_paths": image_paths, "image_descs": image_descs}

    # Save to hidden file in user's home directory
    backup_file = os.path.expanduser("~/.quad_grid_backup.json")
    try:
        with open(backup_file, "w") as f:
            json.dump(backup_data, f)
        print(f"Current selection saved to {backup_file}")
    except Exception as e:
        print(f"Unable to save backup: {e}")


def load_selection():
    """Load previous selection from backup file."""
    backup_file = os.path.expanduser("~/.quad_grid_backup.json")
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
    image_paths = ["", "", "", ""]
    image_descs = ["", "", "", ""]

    # If previous selection exists, ask user whether to use it
    if previous_selection:
        use_previous = input(
            f"Found previous selection (containing {len(previous_selection['image_paths'])} images)\nUse it? (y/n): "
        ).lower()

        if use_previous == "y" or use_previous == "yes":
            image_paths = previous_selection["image_paths"]
            image_descs = previous_selection["image_descs"]

            # Print previous selection
            print("Using previous image selection:")
            for i, (path, desc) in enumerate(zip(image_paths, image_descs)):
                print(f"Image {i+1}: {path}")
                if desc:
                    print(f"  Description: {desc}")

            # Verify files still exist
            missing_files = [
                path for path in image_paths if path and not os.path.exists(path)
            ]
            if missing_files:
                print(
                    f"Warning: The following files do not exist: {', '.join(missing_files)}"
                )
                reselect = input("Reselect all images? (y/n): ").lower()
                if reselect == "y" or reselect == "yes":
                    image_paths = ["", "", "", ""]
                    image_descs = ["", "", "", ""]

    # If not using previous selection, get new input
    if not all(image_paths):
        print("\nPlease specify image files for the quad grid:")
        for i in range(4):
            pos_name = ["Top Left", "Top Right", "Bottom Left", "Bottom Right"][i]

            image_path = input(f"Enter path for {pos_name} image: ")
            while image_path and not os.path.exists(image_path):
                print(f"Error: File '{image_path}' does not exist")
                image_path = input(
                    f"Re-enter path for {pos_name} image (leave empty to skip): "
                )

            image_paths[i] = image_path if image_path else ""

            if image_path:
                include_desc = input(
                    f"Add description for {pos_name} image? (y/n): "
                ).lower()
                if include_desc == "y" or include_desc == "yes":
                    image_descs[i] = input(f"Enter description for {pos_name} image: ")
                else:
                    image_descs[i] = ""
            else:
                image_descs[i] = ""

    # Save current selection for future use
    save_selection(image_paths, image_descs)

    # Create visualization
    visualize_quad_grid(image_paths, image_descs)


def visualize_quad_grid(image_paths, image_descs):
    # Configure academic paper style - enable LaTeX rendering
    try:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
                "font.serif": ["Times", "Times New Roman", "CMU Serif"],
                "font.size": 9,
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
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
                "font.size": 9,
                "axes.linewidth": 0.5,
                "xtick.major.width": 0.5,
                "ytick.major.width": 0.5,
            }
        )

    plt.style.use("seaborn-v0_8-whitegrid")

    # Check if there are any description texts
    has_any_desc = any(desc for desc in image_descs)

    # Create a figure with appropriate proportions
    fig_size = 6  # Base figure size (inches)

    # If there are descriptions, adjust figure proportions
    if has_any_desc:
        fig_height = fig_size * 1.15  # Increase height to accommodate descriptions
    else:
        fig_height = fig_size

    fig = plt.figure(figsize=(fig_size, fig_height))

    # Use unified border color for quad grid
    border_color = "#444444"

    # Calculate ratio of description text and image area
    desc_height_ratio = 0.1  # Description proportion of total height

    # Set image grid positions - considering whether descriptions exist
    if has_any_desc:
        # Reserve space below images for descriptions
        image_height = (1.0 - desc_height_ratio) / 2  # Height per image
        grid_positions = [
            [0, 0.5, 0.5, image_height],  # Top left [left, bottom, width, height]
            [0.5, 0.5, 0.5, image_height],  # Top right
            [0, 0.5 - image_height, 0.5, image_height],  # Bottom left
            [0.5, 0.5 - image_height, 0.5, image_height],  # Bottom right
        ]

        # Description text positions (below respective images)
        desc_positions = [
            [
                0,
                0.5 - desc_height_ratio,
                0.5,
                desc_height_ratio,
            ],  # Top left description
            [
                0.5,
                0.5 - desc_height_ratio,
                0.5,
                desc_height_ratio,
            ],  # Top right description
            [
                0,
                0.5 - image_height - desc_height_ratio,
                0.5,
                desc_height_ratio,
            ],  # Bottom left description
            [
                0.5,
                0.5 - image_height - desc_height_ratio,
                0.5,
                desc_height_ratio,
            ],  # Bottom right description
        ]
    else:
        # Standard 2x2 grid when no descriptions
        grid_positions = [
            [0, 0.5, 0.5, 0.5],  # Top left
            [0.5, 0.5, 0.5, 0.5],  # Top right
            [0, 0, 0.5, 0.5],  # Bottom left
            [0.5, 0, 0.5, 0.5],  # Bottom right
        ]
        desc_positions = [None] * 4

    # Add images to the figure
    for i, (image_path, desc) in enumerate(zip(image_paths, image_descs)):
        # Create image subplot
        ax = fig.add_axes(grid_positions[i])

        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                ax.imshow(np.array(img))
                # Add border
                for spine in ax.spines.values():
                    spine.set_color(border_color)
                    spine.set_linewidth(0.8)
            except Exception as e:
                print(f"Unable to load image {image_path}: {e}")
                ax.text(0.5, 0.5, "Error", ha="center", va="center", fontsize=10)
        else:
            # If no image or image doesn't exist
            ax.set_facecolor("#f0f0f0")  # Light gray background
            ax.text(0.5, 0.5, "No Image", ha="center", va="center", fontsize=10)

        # Remove axes and grid
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

        # Eliminate all internal padding
        ax.set_xmargin(0)
        ax.set_ymargin(0)

        # If there's a description, add description text below the image
        if desc and has_any_desc:
            # Create axis for description text
            desc_ax = fig.add_axes(desc_positions[i])
            desc_ax.axis("off")

            # Format description text using LaTeX
            if plt.rcParams["text.usetex"]:
                desc_text = r"\textrm{" + desc + "}"
            else:
                desc_text = desc

            # Add description in the text axis
            desc_ax.text(0.5, 0.5, desc_text, ha="center", va="center", fontsize=9)

    # Get appropriate output path (current directory)
    output_dir = os.getcwd()

    # Save as PDF and PNG, using compact layout
    output_path_pdf = os.path.join(output_dir, "quad_grid.pdf")
    plt.savefig(
        output_path_pdf, dpi=300, bbox_inches="tight", pad_inches=0, format="pdf"
    )
    print(f"Visualization saved to: {output_path_pdf}")

    output_path_png = os.path.join(output_dir, "quad_grid.png")
    plt.savefig(output_path_png, dpi=300, bbox_inches="tight", pad_inches=0)
    print(f"Visualization also saved as PNG: {output_path_png}")

    # Display image
    plt.show()


if __name__ == "__main__":
    main()
