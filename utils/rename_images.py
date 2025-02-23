import os
import argparse


def rename_images_with_prefix(root_dir, prefix):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
                old_path = os.path.join(root, file)
                new_filename = f"{prefix}_{file}"
                new_path = os.path.join(root, new_filename)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename images with a prefix")
    parser.add_argument("--root_dir", help="Root directory to search for images")
    parser.add_argument("--prefix", help="Prefix to add to image filenames")
    args = parser.parse_args()

    rename_images_with_prefix(args.root_dir, args.prefix)
