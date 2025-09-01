#!/usr/bin/env python3
"""
Process coco_train_ITSDT.txt to count targets per image
Format: image_path target_count
"""

def process_coco_file(input_file, output_file):
    """Process COCO format file to count targets per image"""

    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Split by spaces
        parts = line.split()

        if len(parts) == 0:
            continue

        image_path = parts[0]

        # Count bounding boxes (each bbox has 5 comma-separated values)
        # The format is: x1,y1,x2,y2,class
        # So if we have additional parts beyond the image path, each represents a bbox
        bbox_count = len(parts) - 1  # Subtract 1 for the image path

        results.append(f"{image_path} {bbox_count}")

    # Write results to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(result + '\n')

    print(f"Processed {len(results)} images")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    input_file = "coco_train_DAUB.txt"
    output_file = "Num_DAUB_train.txt"

    process_coco_file(input_file, output_file)
