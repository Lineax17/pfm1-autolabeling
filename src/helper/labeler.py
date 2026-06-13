from ultralytics import YOLO

class Labeler:
    def label_videos(self, video_folder, target_folder, model):
        for video in video_folder.glob("*.mp4"):
            output_name = video.stem

            model(
                str(video),
                save_txt=True,
                save_conf=True,
                save_frames=True,
                save=True,
                project=str(target_folder),
                name=output_name,
                exist_ok=True,
            )