./Source contains the source code files for this project, written in Python with ❤
    There're four configuration files here. You can change the configuration according to that
    There's documentation for the configuration in the loadConfig method of `eigenface.py`
    The files you should use to run stuff are `test.py` and `train.py`
    You can train a model using: 
        `python train.py -p <dataset_path> -i <image_ext, like .jpg> -t <text_ext, like .txt> -c <configuration_file> -m <output_model>`
        Refer to `python train.py -h` for more information
    You can test with a model using: 
        `python test.py -i <test_image> -m <trained_model> -c <configuration_file> -o <output_image>`
        Refer to `python test.py -h` for more information
    And there's also two haar_cascade eye recognizor file. Make sure the program can find them.

./Dataset contains two datasets of our own "MyDataSet" and "MySmallDataSet", roughly 445 images in "MyDataSet" andd 45 images in "MySmallDataSet". All annotated with a text file.

./Test contains some test image. I'm xz and there's two images of me for test. "image_xz_0002.jpg" is NOT in the dataset for training. And "image_xz_0006.jpg" is in the dataset for training.

./Results contains some test results

./Report contains the source and the pdf version of the project report.
    If you find the rendered version overwhelming to read, please double click on `report.md` to open the markdown file and read it directly