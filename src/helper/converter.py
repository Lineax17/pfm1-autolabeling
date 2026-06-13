import json
from pathlib import Path
import cv2

class Converter:
    def get_video_dimensions(self, folder: Path) -> tuple[int, int]:
        avi_files = list(folder.glob("*.avi"))
        if not avi_files:
            raise FileNotFoundError(f"No .avi file found in {folder}")
        cap = cv2.VideoCapture(str(avi_files[0]))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return w, h

    def convert(self, source_folder, target_folder, class_names, file_name, conf_threshold):
        coco = {
            "licenses": [{"name": "", "id": 0, "url": ""}],
            "info": {
                "contributor": "",
                "date_created": "",
                "description": "",
                "url": "",
                "version": "",
                "year": "",
            },
            "categories": [
                {"id": i + 1, "name": name, "supercategory": ""}
                for i, name in enumerate(class_names)
            ],
            "images": [],
            "annotations": [],
        }

        image_id = 1
        annotation_id = 1

        for recording_dir in sorted(source_folder.iterdir()):
            if not recording_dir.is_dir():
                continue

            label_dir = recording_dir / "labels"
            if not label_dir.exists():
                continue

            w, h = source_folder.get_video_dimensions(recording_dir)

            for label_file in sorted(label_dir.glob("*.txt"), key=lambda f: int(f.stem.rsplit("_", 1)[-1])):
                frame_number = int(label_file.stem.rsplit("_", 1)[-1]) * 10
                frame_name = f"frame_{frame_number:06d}.png"

                coco["images"].append({
                    "id": image_id,
                    "width": w,
                    "height": h,
                    "file_name": frame_name,
                    "license": 0,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": 0,
                })

                with open(label_file) as f:
                    for line in f:
                        parts = list(map(float, line.strip().split()))
                        if not parts:
                            continue

                        if len(parts) == 6:
                            cls, xc, yc, bw, bh, conf = parts
                            if conf < conf_threshold:
                                continue
                        else:
                            cls, xc, yc, bw, bh = parts

                        x = (xc - bw / 2) * w
                        y = (yc - bh / 2) * h
                        abs_bw = bw * w
                        abs_bh = bh * h

                        coco["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": int(cls) + 1,
                            "segmentation": [],
                            "area": abs_bw * abs_bh,
                            "bbox": [x, y, abs_bw, abs_bh],
                            "iscrowd": 0,
                            "attributes": {
                                "occluded": False,
                                "rotation": 0.0,
                                "track_id": 0,
                                "keyframe": True,
                            },
                        })
                        annotation_id += 1

                image_id += 1

        target_folder.mkdir(parents=True, exist_ok=True)
        output_path = target_folder / file_name
        with open(output_path, "w") as f:
            json.dump(coco, f, indent=2)
        print(f"Saved {annotation_id - 1} annotations across {image_id - 1} frames → {output_path}")
