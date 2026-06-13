from pathlib import Path
import cv2
import json
from ultralytics import YOLO


class CVATLabeler:
    def label_videos(self, video_folder, target_folder, model_path, frame_stride=10):
        model = YOLO(str(model_path))

        video_folder = Path(video_folder)
        target_folder = Path(target_folder)

        for video in video_folder.glob("*.mp4"):

            video_name = video.stem

            # ----------------------------
            # CVAT structure
            # ----------------------------
            dataset_dir = target_folder / video_name
            images_dir = dataset_dir / "images" / "default"
            annotations_dir = dataset_dir / "annotations"

            images_dir.mkdir(parents=True, exist_ok=True)
            annotations_dir.mkdir(parents=True, exist_ok=True)

            coco_path = annotations_dir / "instances_default.json"

            images = []
            annotations = []
            categories = []

            annotation_id = 0
            image_id = 0

            # ----------------------------
            # categories from model
            # ----------------------------
            for cls_id, name in model.names.items():
                categories.append({
                    "id": cls_id,
                    "name": name
                })

            # ----------------------------
            # video capture
            # ----------------------------
            cap = cv2.VideoCapture(str(video))
            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # stride skipping
                if frame_idx % frame_stride != 0:
                    frame_idx += 1
                    continue

                # CVAT-style filename
                frame_name = f"frame_{frame_idx:06d}.png"
                frame_path = images_dir / frame_name

                cv2.imwrite(str(frame_path), frame)

                h, w = frame.shape[:2]

                images.append({
                    "id": image_id,
                    "file_name": f"images/default/{frame_name}",
                    "width": w,
                    "height": h
                })

                # inference
                results = model(frame, verbose=False)[0]

                if results.boxes is not None:
                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        cls = int(box.cls[0])

                        bw = x2 - x1
                        bh = y2 - y1

                        annotations.append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": cls,
                            "bbox": [x1, y1, bw, bh],
                            "area": bw * bh,
                            "iscrowd": 0
                        })

                        annotation_id += 1

                image_id += 1
                frame_idx += frame_stride

            cap.release()

            coco = {
                "images": images,
                "annotations": annotations,
                "categories": categories
            }

            with open(coco_path, "w") as f:
                json.dump(coco, f, indent=2)

            print(f"✔ CVAT dataset ready: {video_name}")
            print(f"   → {dataset_dir}")