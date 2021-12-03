/*
 * Author : Cédric Fontaine
 * Version : 1.0
 * Date : 19/11/21
 * Licence : MIT
 * 
 * 
 * This program is a school project.
 * Using a computer vision library, it is supposed to detect faces and recognize up to 4 people whose names are hard coded in the recognition function.
 * It uses the OpenCV lib (with the contrib module for the face recognition).
 * I used the Eigen Face Trainer to create a signature for each of my 4 models.
 * And The Fisher Face Recognizer to recognize them on different sample images.
 * The face detection itself is made using the HaarCascade preset provided by OpenCV for front faces and the lib's native detection method.
 * 
 * It is far from perfect but does work on hand selected samples.
 */

// OpenCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>

// OpenCV Contrib
#include <opencv2/face.hpp>
#include <opencv2/face/facerec.hpp>

// Native C
#include <filesystem>
#include <iostream>

using namespace cv::face;
using namespace cv;
using namespace std;

namespace fs = std::filesystem;

// Some global variables for an easier path navigation

string DATA = "./data/"; // data source folder
string MODELS = DATA + "Models/Source/"; // Contains the signatures source folders
string SIGNATURES = DATA + "Models/Dest/"; // Contains the collated signatures for the trainer
string CASCADE = MODELS + "haarcascade.xml"; // The default frontalface haarcascade features file provided by OpenCV
string PHOTOS = DATA + "Photos/"; // The folder containing the samples. Not used.
string EIGEN = DATA + "eigen.yml"; // The file to store the training data into

CascadeClassifier cascade;

// This sets the minimal size of our rectangles of interest.
// It prevents the model from wrongly detecting objects,
// but at the cost of missing objects that would be too small.
int side = 128;
Size SIZE = Size(side, side);

/*
 * This function creates a label on the cropped faces depending on the prefix name
 * 
 * @param images - an empty vector that receives the cropped models as they are found and iterated over.
 * @param label  - an empty vector that receives the models names as they are found and iterated over.
 */
static void labelize(vector<Mat>& images, vector<int>& labels) {
	cout << "[FaceRecognizer] Creating a label for all the extracted model faces" << endl;

	vector<String> modelDir;

	// Stores all the subsequent files names in the cropped models directory.
	// Each folder being a label ranging from 1 to 4
	// The files (our training images) are named like so : label-index.jpg
	// (label and index being numbers)
	glob(SIGNATURES, modelDir, false);

	// For each file
	for (size_t i = 0; i < modelDir.size(); i++)
	{
		string name = "";
		// We extract the label from the image's name
		// Example : data/Models/Dest/12-4.jpg (so the 4th image previously found in the folder named "12")
		// We want the label ID ("12). In our string, it goes from the last path separator ('\') to the last '-'.
		// So from character 16 to character 18.
		// We then need a subsequence between these position to get our label
		size_t nameIndexStart = modelDir[i].rfind('\\', modelDir[i].length());
		size_t nameIndexEnd = modelDir[i].find_last_of('-', modelDir[i].length());
		if (nameIndexStart != string::npos)
		{
			name = modelDir[i].substr(nameIndexStart + 1, nameIndexEnd - nameIndexStart - 1);
		}
		// Both vectors end up with exactly the same size
		// Each image has the same position index as its related label, so the trainer knows how to name our objects.
		images.push_back(imread(modelDir[i], 0));
		labels.push_back(atoi(name.c_str()));
	}
}

/*
 * This function creates an Eigen recognizer model and saves the training results in a data file.
 */
void eigenFaceTrainer() {
	vector<Mat> images;
	vector<int> labels;
	labelize(images, labels);

	cout << "[FaceRecognizer] Starting the Eigen Face Recognizer training" << endl;

	Ptr<EigenFaceRecognizer> eigen = EigenFaceRecognizer::create();
	//Ptr<LBPHFaceRecognizer> lbph = LBPHFaceRecognizer::create(1, 8, 8, 8);

	eigen->train(images, labels);

	eigen->save(EIGEN);

	cout << "[FaceRecognizer] Training complete. Results have been stored in " << EIGEN << endl;
}

/*
 * Iterate over the models folder and assign a label for each of them.
 * The program basically just crops the faces in the images and give them a name with the following syntax:
 * <label>-<index>.jpg
 * label being the folder's reference (1 to 4 to n)
 * index being the image position in this folder (eg: 1-5 is the 5th signature image of the 1st object)
 */
void createSignatures()
{
	cout << "[FaceRecognizer] Starting the creation of the faces' signatures" << endl;

	vector<Rect> faces;
	string path;
	string name;
	Mat frame;
	Mat gray;
	Mat crop;
	Mat res;

	int count;
	int i;

	path = MODELS;

	if (fs::exists(path))
	{
		for (const auto& id : fs::directory_iterator(path))
		{
			name = id.path().filename().string();
			if (!name.ends_with(".xml"))
			{
				cout << "[FaceRecognizer] Parsing the faces in the '" << name << "' directory." << endl;
				count = 0;
				for (const auto& image : fs::directory_iterator(id.path()))
				{
					frame = imread(image.path().string());
					cvtColor(frame, gray, COLOR_BGR2GRAY);
					equalizeHist(gray, gray);

					cascade.detectMultiScale(gray, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, SIZE);

					Rect faceSignature;

					for (i = 0; i < faces.size(); i++)
					{
						faceSignature.x = faces[i].x;
						faceSignature.y = faces[i].y;
						faceSignature.width = (faces[i].width);
						faceSignature.height = (faces[i].height);

						crop = frame(faceSignature);
						resize(crop, res, SIZE, 0, 0, INTER_LINEAR);
						cvtColor(crop, gray, COLOR_BGR2GRAY);

						// Saves the cropped result into the model destination folder.
						imwrite(SIGNATURES + name + "-" + to_string(count++) + ".jpg", res);
					}
				}
			}
		}
		eigenFaceTrainer();
	}
}

/*
 * This function scans the image using the Eigen data previously created
 * It will circle all detected faces and count the recognized objects
 */
void scan(string path) {

	cout << "[FaceRecognizer] Scanning the image for faces" << endl;

	Ptr<FaceRecognizer>  model = FisherFaceRecognizer::create();
	//Ptr<FaceRecognizer>  model = LBPHFaceRecognizer::create();
	model->read(EIGEN);

	Point center;

	long count = 0;
	int reco = 0;
	string name = "";

	vector<Rect> faces;
	Mat frame;
	Mat grayImage;
	Mat original;

	frame = imread(path);
		
	if (!frame.empty()) {

		original = frame.clone();

		// Convert the image to gray to get rid of unwanted details
		cvtColor(original, grayImage, COLOR_BGR2GRAY);
		equalizeHist(grayImage, grayImage);

		// Scan the image for faces
		cascade.detectMultiScale(grayImage, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, SIZE);

		for (int i = 0; i < faces.size(); i++)
		{
			// region of interest containing the face
			Rect faceROI = faces[i];

			// crop the gray image to extract the face only
			Mat face = grayImage(faceROI);

			// resizing the cropped image to fit the database image sizes
			Mat resized_face;
			resize(face, resized_face, SIZE, 1.0, 1.0, INTER_CUBIC);

			// Call to the Fisher's prediction method
			int label;
			double confidence;
			model->predict(resized_face, label, confidence);

			string text = name;

			// this sets a minimum threshold to prevent the model from wrongly assigning a face to the closest signature
			if (confidence < 1500) label = 0;
			
			// counts the recognized faces
			if (label) reco++;

			// Not the most optimized way, but surely the easiest one that worked when trying to associate a label number to a name
			switch (label) {
				case 0:
					name = "Unknown";
					break;
				case 1:
					name = "Cedric";
					break;
				case 2:
					name = "Kim Jong Un";
					break;
				case 3:
					name = "Jeff Bezos";
					break;
				case 4:
					name = "Morgan Freeman";
					break;
			}

			cout << "[FaceRecognizer] Confidence ratio: " << confidence << " - Signature label: " << label << " - Name: " << name << endl;

			// visual information
			center = Point(faces[i].tl().x + faces[i].width / 2, faces[i].tl().y + faces[i].height / 2);
			circle(original, center, faces[i].width / 2, Scalar(0, 255, 0), 2);
			putText(original, name, Point(faces[i].tl().x + faces[i].width*0.75, faces[i].tl().y + faces[i].height), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 255, 0), 1);
		}

		putText(original, "Detected : " + to_string(faces.size()), Point(10, 30), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);
		putText(original, "Recognized : " + to_string(reco), Point(10, 50), FONT_HERSHEY_COMPLEX_SMALL, 1.0, CV_RGB(0, 255, 0), 1.0);

		imshow("FaceRecognizer", original);
		waitKey(0);
		destroyAllWindows();
	}
}

/*
 * The core program calls all the subjacent methods.
 */
int main()
{
	string path;
	bool quit = false;

	cascade.load(CASCADE);

	if (cascade.empty())
	{
		cout << "Failed to open the Haar Cascade file at " << CASCADE << endl;
		return EXIT_FAILURE;
	}

	createSignatures();

	do {
		cout << "[FaceRecognizer] Please enter the path (relative or absolute) to your photo : ";
		cin >> path;

		quit = path.compare("quit") == 0;

		if (!quit)
		{
			while (!fs::exists(path))
			{
				cout << "[FaceRecognizer] Error - No photo found at the following location: " << path << "." << endl << "Please select a valid URI." << endl;
				cout << "Example: data/Photos/your_photo.jpg" << endl;
				cin >> path;
			}
			scan(path);
		}
	} while (!quit);

	return EXIT_SUCCESS;
}