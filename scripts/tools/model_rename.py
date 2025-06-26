import os
import torch
import copy
from collections import OrderedDict
import re


def print_colored(text, color="yellow"):
    """Print text in color for better visibility in the terminal."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "end": "\033[0m",
    }
    print(f"{colors.get(color, colors['yellow'])}{text}{colors['end']}")


def get_input_bool(prompt, default=True):
    """Get yes/no input from user with default value."""
    if default:
        choices = "[Y/n]"
    else:
        choices = "[y/N]"

    response = input(f"{prompt} {choices}: ").strip().lower()
    if response in ["", "y", "yes"]:
        return True
    elif response in ["n", "no"]:
        return False
    else:
        print_colored("Invalid input, using default.", "red")
        return default


def display_dict_structure(d, prefix="", max_items=10, max_value_len=100):
    """Display dictionary structure with limited items and compact value
    representation."""
    if not isinstance(d, dict):
        value_str = str(d)
        if isinstance(d, torch.Tensor):
            value_str = f"Tensor{tuple(d.shape)} {d.dtype}"
        elif hasattr(d, "__len__") and not isinstance(d, str):
            value_str = f"{type(d).__name__}[{len(d)}]"

        if len(value_str) > max_value_len:
            value_str = value_str[:max_value_len] + "..."
        print(f"{prefix}{value_str}")
        return

    # Recursively print dict structure
    items = list(d.items())
    if len(items) > max_items:
        print(f"{prefix}Dict with {len(items)} keys: (showing first {max_items})")
        items = items[:max_items]
    else:
        print(f"{prefix}Dict with {len(items)} keys:")

    # Display sample keys
    for i, (k, v) in enumerate(items):
        is_last = i == len(items) - 1
        new_prefix = f"{prefix}│   " if not is_last else f"{prefix}    "
        end_char = "├── " if not is_last else "└── "

        # Print the key
        key_str = f"{prefix}{end_char}{k}"
        if isinstance(v, dict):
            print(f"{key_str} →")
            display_dict_structure(v, new_prefix, max_items, max_value_len)
        else:
            value_str = str(v)
            if isinstance(v, torch.Tensor):
                value_str = f"Tensor{tuple(v.shape)} {v.dtype}"
            elif hasattr(v, "__len__") and not isinstance(v, str):
                value_str = f"{type(v).__name__}[{len(v)}]"

            if len(value_str) > max_value_len:
                value_str = value_str[:max_value_len] + "..."
            print(f"{key_str}: {value_str}")


def analyze_key_structure(data):
    """Analyze keys to detect common prefixes and structure patterns."""
    if not isinstance(data, dict) or not data:
        return None

    # Extract all prefixes before the first dot
    prefixes = {}
    for key in data.keys():
        parts = key.split(".")
        if len(parts) > 1:
            prefix = parts[0]
            if prefix in prefixes:
                prefixes[prefix] += 1
            else:
                prefixes[prefix] = 1

    # Sort by frequency
    sorted_prefixes = sorted(prefixes.items(), key=lambda x: x[1], reverse=True)

    if not sorted_prefixes:
        return None

    return sorted_prefixes


def rename_with_smart_prefix(data):
    """Rename keys with intelligent prefix detection and replacement."""
    if not isinstance(data, dict):
        return data

    # Analyze key structure
    prefixes = analyze_key_structure(data)

    if not prefixes:
        print_colored("No common prefixes found in keys.", "yellow")
        return data

    # Show detected prefixes
    print_colored("Detected common prefixes:", "blue")
    for i, (prefix, count) in enumerate(prefixes):
        print(f"{i+1}. '{prefix}' (used in {count} keys)")

    # Ask which prefix to replace
    while True:
        prefix_choice = input(
            "\nSelect prefix to replace (number) or enter custom prefix: "
        ).strip()

        old_prefix = None
        try:
            # Check if it's a number referring to detected prefixes
            choice_idx = int(prefix_choice) - 1
            if 0 <= choice_idx < len(prefixes):
                old_prefix = prefixes[choice_idx][0]
            else:
                print_colored("Invalid selection.", "red")
        except ValueError:
            # Custom prefix entered
            if prefix_choice:
                old_prefix = prefix_choice

        if old_prefix:
            break

    # Get new prefix
    new_prefix = input(f"Enter new prefix to replace '{old_prefix}': ").strip()

    # Find keys with the prefix (either exact match or followed by dot)
    prefix_with_dot = f"{old_prefix}."
    matches = [
        key
        for key in data.keys()
        if key == old_prefix or key.startswith(prefix_with_dot)
    ]

    if not matches:
        print_colored(f"No keys found with prefix '{old_prefix}'", "yellow")
        return data

    # Show preview with full paths
    print_colored(f"Found {len(matches)} keys with prefix '{old_prefix}':", "blue")
    for key in matches[:10]:  # Show first 10
        if key == old_prefix:
            new_key = new_prefix
        else:
            new_key = f"{new_prefix}{key[len(old_prefix):]}"
        print(f"  '{key}' → '{new_key}'")

    if len(matches) > 10:
        print(f"  ... and {len(matches) - 10} more")

    # Confirm
    if get_input_bool("Apply these changes?"):
        # Create new dict to avoid modification during iteration
        new_data = {}
        for key, value in data.items():
            if key == old_prefix:
                new_key = new_prefix
                new_data[new_key] = value
            elif key.startswith(prefix_with_dot):
                new_key = f"{new_prefix}{key[len(old_prefix):]}"
                new_data[new_key] = value
            else:
                new_data[key] = value

        print_colored(f"Renamed {len(matches)} keys", "green")
        return new_data

    return data


def navigate_and_rename(data, path=None):
    """Navigate through nested dictionaries and allow renaming keys."""
    if path is None:
        path = []

    # Copy the dictionary to avoid modifying during iteration
    if isinstance(data, dict):
        current_level = data

        # Display the current location in the nested structure
        path_str = " → ".join(path) if path else "root"
        print_colored(f"\nCurrent location: {path_str}", "blue")

        # Display dictionary structure
        print_colored("Structure:", "cyan")
        display_dict_structure(current_level)

        # Check if there are any dictionaries to navigate into
        has_dict_values = any(isinstance(v, dict) for v in current_level.values())

        # Ask if user wants to rename keys at this level
        if get_input_bool("Rename keys at this level?"):
            current_level_copy = copy.deepcopy(current_level)
            for key in list(current_level_copy.keys()):
                new_key = input(f"Rename '{key}' to (leave empty to keep): ").strip()
                if new_key and new_key != key:
                    current_level[new_key] = current_level.pop(key)
                    print_colored(f"Renamed '{key}' to '{new_key}'", "green")

        # Process nested dictionaries if present
        if has_dict_values:
            for key, value in list(current_level.items()):
                if isinstance(value, dict):
                    if get_input_bool(f"Navigate into '{key}'?"):
                        new_path = path + [key]
                        navigate_and_rename(value, new_path)

        return data
    else:
        print_colored(f"Cannot navigate: not a dictionary", "red")
        return data


def bulk_rename_with_regex(data, pattern=None, replacement=None):
    """Apply regex pattern to rename multiple keys at once."""
    if not isinstance(data, dict):
        return data

    if pattern is None:
        pattern = input("Enter regex pattern to match keys: ")

    if not pattern:
        return data

    if replacement is None:
        replacement = input("Enter replacement pattern: ")

    try:
        regex = re.compile(pattern)

        # First find all matches to show user what will change
        matches = []
        for key in data.keys():
            if regex.search(key):
                new_key = regex.sub(replacement, key)
                if new_key != key:
                    matches.append((key, new_key))

        if not matches:
            print_colored("No keys match the pattern.", "yellow")
            return data

        # Show preview of changes
        print_colored(f"Found {len(matches)} keys to rename:", "blue")
        for old_key, new_key in matches[:10]:  # Show first 10 for preview
            print(f"  '{old_key}' → '{new_key}'")

        if len(matches) > 10:
            print(f"  ... and {len(matches) - 10} more")

        # Confirm changes
        if get_input_bool("Apply these changes?"):
            # Create a new dict to avoid modification during iteration
            new_data = {}
            for key, value in data.items():
                if regex.search(key):
                    new_key = regex.sub(replacement, key)
                    new_data[new_key] = value
                else:
                    new_data[key] = value

            print_colored(f"Applied {len(matches)} renames", "green")
            return new_data
    except re.error as e:
        print_colored(f"Invalid regex pattern: {e}", "red")

    return data


def rename_with_pattern(data, old_prefix=None, new_prefix=None):
    """Rename keys by replacing prefix."""
    if not isinstance(data, dict):
        return data

    if old_prefix is None:
        old_prefix = input("Enter prefix to replace: ")

    if not old_prefix:
        return data

    if new_prefix is None:
        new_prefix = input("Enter new prefix: ")

    # Find keys with the prefix
    matches = [key for key in data.keys() if key.startswith(old_prefix)]

    if not matches:
        print_colored(f"No keys found with prefix '{old_prefix}'", "yellow")
        return data

    # Show preview
    print_colored(f"Found {len(matches)} keys with prefix '{old_prefix}':", "blue")
    for key in matches[:10]:  # Show first 10
        new_key = new_prefix + key[len(old_prefix) :]
        print(f"  '{key}' → '{new_key}'")

    if len(matches) > 10:
        print(f"  ... and {len(matches) - 10} more")

    # Confirm
    if get_input_bool("Apply these changes?"):
        # Create new dict to avoid modification during iteration
        new_data = {}
        for key, value in data.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix) :]
                new_data[new_key] = value
            else:
                new_data[key] = value

        print_colored(f"Renamed {len(matches)} keys", "green")
        return new_data

    return data


def deep_apply_rename(data, rename_func, **kwargs):
    """Apply renaming function to all nested dictionaries."""
    if not isinstance(data, dict):
        return data

    # Apply to current level
    data = rename_func(data, **kwargs)

    # Apply to nested dictionaries
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = deep_apply_rename(value, rename_func, **kwargs)

    return data


def process_recursive_levels(data):
    """Process dictionary recursively with user confirmation at each level."""
    if not isinstance(data, dict):
        return data

    print_colored("\n=== Processing top level ===", "purple")
    display_dict_structure(data)

    # First, offer to rename keys at top level
    while True:
        print_colored("\nRename Options:", "blue")
        print("1. Smart prefix detection and replacement")
        print("2. Manual key-by-key renaming")
        print("3. Regex pattern replacement")
        print("4. Simple prefix replacement")
        print("5. Continue without renaming")

        choice = input("Select an option (1-5): ").strip()

        if choice == "1":
            data = rename_with_smart_prefix(data)
        elif choice == "2":
            data = navigate_and_rename(data)
        elif choice == "3":
            data = bulk_rename_with_regex(data)
        elif choice == "4":
            data = rename_with_pattern(data)
        elif choice == "5":
            break
        else:
            print_colored("Invalid choice, please select 1-5", "red")
            continue

        # Ask if user wants to continue renaming at this level
        if not get_input_bool("Continue renaming at this level?", default=False):
            break

    # Ask about bulk operations across all levels
    if get_input_bool(
        "Perform deep rename operations (across all nested levels)?", default=False
    ):
        while True:
            print_colored("\nDeep Rename Options:", "blue")
            print("1. Deep regex pattern replacement (all levels)")
            print("2. Deep prefix replacement (all levels)")
            print("3. Done with deep operations")

            choice = input("Select an option (1-3): ").strip()

            if choice == "1":
                pattern = input("Enter regex pattern to match keys: ")
                replacement = input("Enter replacement pattern: ")
                if pattern:
                    data = deep_apply_rename(
                        data,
                        bulk_rename_with_regex,
                        pattern=pattern,
                        replacement=replacement,
                    )
            elif choice == "2":
                old_prefix = input("Enter prefix to replace: ")
                new_prefix = input("Enter new prefix: ")
                if old_prefix:
                    data = deep_apply_rename(
                        data,
                        rename_with_pattern,
                        old_prefix=old_prefix,
                        new_prefix=new_prefix,
                    )
            elif choice == "3":
                break
            else:
                print_colored("Invalid choice, please select 1-3", "red")

    return data


def main():
    print_colored("=== Model Checkpoint Renaming Tool ===", "green")
    print(
        "This tool allows you to interactively rename keys in model checkpoint files."
    )

    # Get the input file
    while True:
        ckpt_path = input("Enter path to .ckpt file: ").strip()
        if os.path.exists(ckpt_path) and ckpt_path.endswith(".ckpt"):
            break
        print_colored("Invalid path or not a .ckpt file. Please try again.", "red")

    # Load the checkpoint
    print_colored(f"Loading checkpoint from {ckpt_path}...", "blue")
    try:
        # Add a warning about security when loading checkpoints
        print_colored(
            "SECURITY NOTE: Loading with weights_only=False. For untrusted models, consider using weights_only=True.",
            "yellow",
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        print_colored("Checkpoint loaded successfully.", "green")
    except Exception as e:
        print_colored(f"Error loading checkpoint: {e}", "red")
        return

    # Process the checkpoint
    print_colored("\nTop level keys in checkpoint:", "cyan")
    for i, key in enumerate(checkpoint.keys()):
        print(f"{i+1}. {key}")

    # Let user choose which keys to process
    while True:
        key_idx = input("\nSelect a key to process (number) or 'q' to quit: ").strip()
        if key_idx.lower() == "q":
            break

        try:
            key_idx = int(key_idx) - 1
            if 0 <= key_idx < len(checkpoint.keys()):
                key = list(checkpoint.keys())[key_idx]
                print_colored(f"Processing '{key}'...", "blue")

                # Process the selected part of the checkpoint
                checkpoint[key] = process_recursive_levels(checkpoint[key])
            else:
                print_colored("Invalid selection.", "red")
        except ValueError:
            print_colored("Please enter a number or 'q'.", "red")

    # Save the modified checkpoint
    if get_input_bool("\nSave modified checkpoint?"):
        # Generate output filename
        basename, ext = os.path.splitext(ckpt_path)
        output_path = f"{basename}_renamed{ext}"

        # Allow custom filename
        custom_name = input(f"Enter output filename (default: {output_path}): ").strip()
        if custom_name:
            if not custom_name.endswith(".ckpt"):
                custom_name += ".ckpt"
            output_path = custom_name

        # Save the checkpoint
        print_colored(f"Saving to {output_path}...", "blue")
        try:
            torch.save(checkpoint, output_path)
            print_colored(f"Checkpoint saved successfully to {output_path}", "green")
        except Exception as e:
            print_colored(f"Error saving checkpoint: {e}", "red")
    else:
        print_colored("Changes discarded.", "yellow")


if __name__ == "__main__":
    main()
