# PFM1 Autolabeling Tool

This tool can autolabel footage containing a PFM1 replica.

## Usage

1. Clone the repository and navigate to the project directory.

2. Install the required dependencies using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your footage in the data/videos directory.

4. Run the autolabeling script:

   ```bash
   python src/main.py
   ```
5. The autolabeled footage will be saved in the data/output directory.

## Testing

To test your labeled footage, you can use the provided test script:

- In the configuration section adjust the video name (without the .mp4).
- Run the test script:

   ```bash
   python src/test/test_show_bbox.py
   ```
  
If you should use the COCOLabeler instead of the default you have to adjust the paths.

## Details

The Project contains two different ways to label data:

- CVATLabeler (Default): Outputs CVAT-like COCO annotations. This is the default way to label data. 
This is used throughout my PFM1 Project and can directly be injected into my Transformation Pipeline. 
Use this if you want to contribute to my PFM1 Dataset.
- COCOLabeler: Outputs COCO annotations. This should be used if you start from scratch and don't have to deal with CVAT.

## License

The project ships a custom YOLOv11 model. Because of that the project does inherit the AGPLv3 License of the Ultralytics YOLOv11 model.
