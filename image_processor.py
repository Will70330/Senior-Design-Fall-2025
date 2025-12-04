import cv2
import os
import shutil
import numpy as np
from pathlib import Path
import re # For regex to parse filenames
import json

class ImageProcessor:
    def __init__(self):
        pass

    def calculate_blur_score(self, image):
        """Calculate the Laplacian variance of the image (higher is sharper)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def process_images(self, image_dir, max_frames=200, min_frames=50):
        """
        Analyzes images in image_dir, keeps the best ones, and moves rejected ones
        to a 'rejected' subdirectory.
        """
        image_dir = Path(image_dir)
        images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        
        if not images:
            return 0, 0

        print(f"Analyzing {len(images)} images...")
        
        scored_images = []
        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                score = self.calculate_blur_score(img)
                scored_images.append((score, img_path))
            except Exception as e:
                print(f"Error reading {img_path}: {e}")

        # Sort by score (descending)
        scored_images.sort(key=lambda x: x[0], reverse=True)

        # Determine cutoff
        total_valid = len(scored_images)
        
        if total_valid <= max_frames:
            # If we have fewer than max, keep all valid ones (or filter extremely blurry ones if we had a threshold)
            # For now, we just keep them all if under the limit, unless we want to enforce quality.
            # Let's enforce a loose quality check: drop the bottom 10% if they are significantly worse? 
            # Simpler: just keep all if under max, to ensure coverage.
            keep_count = total_valid
        else:
            # Keep top max_frames
            keep_count = max_frames

        # Ensure we don't drop below min_frames if possible
        keep_count = max(keep_count, min(min_frames, total_valid))
        
        to_keep = set(img_path for _, img_path in scored_images[:keep_count])
        
        # Move rejected
        rejected_dir = image_dir.parent / "rejected_images"
        rejected_dir.mkdir(exist_ok=True)
        
        rejected_count = 0
        for _, img_path in scored_images:
            if img_path not in to_keep:
                try:
                    shutil.move(str(img_path), str(rejected_dir / img_path.name))
                    rejected_count += 1
                except Exception as e:
                    print(f"Error moving {img_path}: {e}")

        print(f"Processing complete. Kept {keep_count}, Rejected {rejected_count}")
        return keep_count, rejected_count

    def rename_and_reindex_images(self, image_dir):
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            print(f"Error: {image_dir} is not a valid directory.")
            return False, "Error: Image directory not found."

        print(f"Renaming and re-indexing images in: {image_dir}")

        files_map = {}
        # Regex to find the LAST number in the filename (e.g. IMG_001.jpg -> 001)
        number_pattern = re.compile(r"(\d+)(?!.*\d)") 

        found_any = False

        for f in image_dir.iterdir():
            if not f.is_file(): continue
            if f.name.startswith("."): continue
            
            lower_name = f.name.lower()
            
            # Determine type
            is_depth = "depth" in lower_name
            is_image = lower_name.endswith(('.jpg', '.jpeg', '.png')) and not is_depth
            
            if not (is_depth or is_image):
                continue
                
            # Extract index
            match = number_pattern.search(f.name)
            if match:
                idx = int(match.group(1))
            else:
                continue 
            
            if idx not in files_map:
                files_map[idx] = {}
            
            if is_depth:
                files_map[idx]['depth'] = f
            else:
                files_map[idx]['color'] = f
                files_map[idx]['ext'] = f.suffix
                found_any = True

        if not found_any:
            return False, "No valid image files found (looking for digit-containing .jpg/.png)."

        # Sort by the extracted index
        sorted_indices = sorted(files_map.keys())
        
        renamed_count = 0
        errors = 0
        
        # Track name changes for transforms.json
        # { old_name: new_name }
        rename_map = {}

        # Re-index
        for new_idx, old_idx in enumerate(sorted_indices):
            entry = files_map[old_idx]
            
            # Handle Color
            if 'color' in entry:
                old_path = entry['color']
                ext = entry['ext']
                new_name = f"frame_{new_idx:05d}{ext}"
                new_path = image_dir / new_name
                
                if new_path != old_path:
                    try:
                        old_path.rename(new_path)
                        rename_map[old_path.name] = new_name
                        renamed_count += 1
                    except Exception as e:
                        print(f"Failed to rename {old_path.name}: {e}")
                        errors += 1
            
            # Handle Depth
            if 'depth' in entry:
                old_path = entry['depth']
                new_name = f"depth_{new_idx:05d}.png"
                new_path = image_dir / new_name
                if new_path != old_path:
                    try:
                        old_path.rename(new_path)
                        # We don't usually track depth in transforms.json but good to be consistent if needed
                    except Exception as e:
                        print(f"Failed to rename {old_path.name}: {e}")

        # Check and update transforms.json
        json_path = image_dir.parent / "transforms.json"
        json_updated = False
        
        if json_path.exists() and rename_map:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if 'frames' in data:
                    count = 0
                    for frame in data['frames']:
                        fp = frame.get('file_path', '')
                        # fp is usually "images/name.jpg"
                        path_parts = fp.split('/')
                        filename = path_parts[-1]
                        
                        if filename in rename_map:
                            new_filename = rename_map[filename]
                            # Reconstruct path
                            path_parts[-1] = new_filename
                            frame['file_path'] = "/".join(path_parts)
                            count += 1
                    
                    if count > 0:
                        with open(json_path, 'w') as f:
                            json.dump(data, f, indent=4)
                        print(f"Updated {count} paths in transforms.json")
                        json_updated = True
                        
            except Exception as e:
                print(f"Error updating transforms.json: {e}")

        msg = f"Successfully renamed {renamed_count} images."
        if json_updated:
            msg += " Updated transforms.json."
        elif json_path.exists():
            msg += " transforms.json found but no matching entries updated (mismatch?)."

        if errors > 0:
             return False, f"Renamed {renamed_count} images, but had {errors} errors."
        
        return True, msg

