## What does this repository contain?
This repository contains a code to classify wheather a person is wearing a face mask or not.
This involves the following steps:

	1- First it detects face using Facenet (using a predefined model)
		This is better than Haar Cascade since it also detects faces of people wearing glasses and hence is more accurate

	2- Then in the region of interest it applies the trained mask model to see if the person is wearing a mask or not.

Please note: The dataset was originally created using facial landmarks.

	
The model used is Mobilenetv2 with fine tuning on its head.

I have also used command line arguments for referring various paths in the code. Please refer to references for more information.
The dataset used has 2 folders - Faces with and without mask containging 600+ images each.



## How to use this repository?
Simply clone the repo and install requirements in your virtual environment.
	pip install -r requirements.txt

If you wish to retrain, you may edit the code "model_training.py" and run it.
Otherwise, you can simply run "runme.py"

You may add your own dataset and refer it using command line arguments.





## References and Acknowledgements
A big thanks to Adrian Rosebrock for guiding me through the process.
A big thanks to Prajna Bhandary for the dataset.

https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/
https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/
https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/
https://www.pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/


