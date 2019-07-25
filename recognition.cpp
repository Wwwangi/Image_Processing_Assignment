
//Import libraries
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <iterator>
#include<valarray>

//avoid using redundant "cv"s and "std"s throughout the program
using namespace cv;
using namespace std;

//Move the image along the x-axis and y-axis
Mat translateImg(Mat& img, int offsetx, int offsety) {
	Mat trans_mat = (Mat_<double>(2, 3) << 1, 0, offsetx, 0, 1, offsety);
	warpAffine(img, img, trans_mat, img.size());
	return img;
}

int main() {
	int i, j, k, width = 200, height = 250, p = 1, q = 1;   //The size of block is defined using p and q so that it can be adjusted easily for comparison purposes
	cout << "The size of block: " + to_string(p) + "*" + to_string(q) << endl <<endl;
	int black = 0, sum1 = 0, sum2 = 0, count = 0, det = 0, shift=0;
	int size = (width / p) * (height / q);  //The dimension of the feature vector (One block for one element)
	array<string, 6>labels{ "世", "界", "一" ,"番","可","爱" }; //Six Chinese characters to be recognized

	Mat image[36];   //Training data
	Mat image_binary[36]; //Training data that are converted into binary images
	vector<vector<double>>features(36);    //A vector that has 36 entries (feature vectors) for 36 training images
	vector<vector<double>>features_centroid(6);   //The centroid feature vector for six patterns

	//Assign size to the feature vector
	for (i = 0; i < 36; i++) {
		features[i].resize(size);
		if (i < 6) {
			features_centroid[i].resize(size);
		}
	}
	//Training
	for (k = 0; k < 36; k++) {  //Process training images one by one
		black = 0, sum1 = 0, sum2 = 0, count = 0;
		image[k] = imread(to_string(k + 1) + ".png", 1);  //Read images
		cvtColor(image[k], image[k], COLOR_RGB2GRAY);  //Convert to grayscale images
		threshold(image[k], image_binary[k], 100, 255, THRESH_BINARY);  //Convert to binary image
		resize(image_binary[k], image_binary[k], Size(width, height));  //Pre-process, to resize the images so that they are more easily to be learned
		//imshow(labels[det] + to_string(shift + 1), image_binary[k]);

		//Count the black pixels in every block and store the values in feature vectors
		for (i = 0; i < image_binary[k].rows; i++) {
			for (j = 0; j < image_binary[k].cols; j++) {
				if ((int)image_binary[k].at<uchar>(i, j) == 0) {  //Iteratively read the pixels, "0" means black pixel in a binary image
					black++;
				}
				sum2++;  //To detect the block boundary in columes
				if (sum2 == p) {  //Change a block
					sum2 = 0;
					features[k][count] += black;
					black = 0;
					count++;
					//cout << count <<endl;
				}
			}
			sum1++;  //To detect the block boundary in rows
			count -= int(width / p);
			if (sum1 == q) {  //Change a block
				count += int(width / p);
				sum1 = 0;
			}
		}

		for (int m = 0; m < size; m++) {
			features[k][m] = double(features[k][m] / p / q);  //Calculate the ratio of black pixels in one block
			features_centroid[det][m] += features[k][m];  //Add the value to centroid feature vector for later averaging process
			//cout <<features_centroid[i]<<endl;
		}
		//cout << det;
		shift++;  //"Shift" is used for the change of patterns from one character to another character
		if (shift == 6) {  //5 images for one character
			//cout << shift << endl;
			det++;
			shift = 0;
		}
	}

	//Calculate the feature centroid vector
	for (int all = 0; all < 6; all++) {
		for (int n = 0; n < size; n++) {
			features_centroid[all][n] = double(features_centroid[all][n] / 6);  //Take the average of 5 samples
		}
	}


	//Test by hand-written characters
	/*
	Mat TestImage,TestImage_binary,TestImage_modified; 
	vector<double>Feature_for_test(size);  
	vector<double>Feature_for_test_modified(size);
	TestImage = imread("ai_small3.png", 1);
	cvtColor(TestImage, TestImage, COLOR_RGB2GRAY);
	threshold(TestImage, TestImage_binary, 100, 255, THRESH_BINARY);
	resize(TestImage_binary, TestImage_binary, Size(width, height));
	//blur(TestImage_binary, TestImage_binary, Size(3, 3));
	imshow("Image for classification", TestImage_binary);
	int left, right, up, down,pass=0;
	int left_pixel, right_pixel, up_pixel, down_pixel;
	int move_vertical, move_horizontal;

	//Pre-processing
	//Adjust the position of the testing image so that it is in the center 
	int first=0;
	for (i = 0; i < TestImage_binary.rows; i++) {
		for (j = 0; j < TestImage_binary.cols; j++) {
			if ((int)TestImage_binary.at<uchar>(i, j) == 0 && first == 0) {
				first = 1;
				up_pixel = i;   //Find the first row that contains the black pixel (upper boundary)
			}
			if (first == 1) {
				break;
			}
		}
		if (first == 1) {
			break;
		}
	}
	//cout << up_pixel << endl;

	first = 0;
	for (i = TestImage_binary.rows-1; i >=0; i--) {
		for (j = TestImage_binary.cols - 1; j >= 0; j--) {
			if ((int)TestImage_binary.at<uchar>(i, j) == 0 && first == 0) {
				first = 1;
				down_pixel = height-i;  //Find the last row that contains the black pixel (lower boundary)
			}
			if (first == 1) {
				break;
			}
		}
		if (first == 1){
			break;
		}
	}
	//cout << down_pixel << endl;
	
	int first_appear_left[250],first_appear_right[250];
	for (i = 0; i < TestImage_binary.rows; i++) {
		first = 0;
		for (j = 0; j < TestImage_binary.cols; j++) {
			if ((int)TestImage_binary.at<uchar>(i, j) == 0 && first==0) {
				first = 1;
				first_appear_left[i] = j;  // Find the black pixel that first appears in all rows, The first_appear_left array contains all the indexes (the left boundary)
			}
			if (first == 0) {
				first_appear_left[i] = 1000;
			}
		}
		//cout << first_appear_left[i] << endl;
	}

	for (i = 0; i < TestImage_binary.rows; i++) {
		first = 0;
		for (j = TestImage_binary.cols-1; j >=0 ; j--) {
			if ((int)TestImage_binary.at<uchar>(i, j) == 0 && first==0) {
				first = 1;
				first_appear_right[i] = j;  //Find the black pixel that last appears in all rows, The first_appear_right array contains all the indexes (the right boundary)
			}
			if (first == 0) {
				first_appear_right[i] = 0;
			}
		}
		//cout << first_appear_right[i] << endl;
	}

	left = distance(first_appear_left, min_element(first_appear_left, first_appear_left + 250));  //Find the row that has earliest black pixel appearence on the left
	left_pixel = first_appear_left[left];
	//cout << left_pixel << endl;
	right = distance(first_appear_right, max_element(first_appear_right, first_appear_right + 250));  //Find the row that has earliest black pixel appearence on the right
	right_pixel = width-first_appear_right[right];
	//cout << right_pixel << endl;

	
	//Move the character along the vertical axis
	int direction_vertical;
	move_vertical = (up_pixel + down_pixel) / 2;    //Make the number of white pixels above and below the character equal
	if (up_pixel >= down_pixel) {
		move_vertical = up_pixel - move_vertical;
		direction_vertical = -1;
	}
	else {
		move_vertical = move_vertical - up_pixel;
		direction_vertical = 1;
	}

	//cout << move_vertical<<endl;
	//cout << direction_vertical << endl;

	//Move the character along the horizontal axis
	int direction_horizontal;
	move_horizontal = (left_pixel + right_pixel) / 2;  //Make the number of white pixels on the left and right side of the character equal
	if (left_pixel >= right_pixel) {
		move_horizontal = left_pixel - move_horizontal;
		direction_horizontal = -1;
	}
	else {
		move_horizontal = move_horizontal - left_pixel;
		direction_horizontal = 1;
	}

	//cout << move_horizontal << endl;
	//cout << direction_horizontal << endl;

	//Move the character so that it is in the center of the image
	translateImg(TestImage_binary, direction_horizontal*move_horizontal, direction_vertical* move_vertical);
	
	//To repaint the black pixels that are generated by the moving the character to white pixels
	if (direction_vertical == -1) {
		for (i = TestImage_binary.rows - 1; i > TestImage_binary.rows - move_vertical - 1; i--) {
			for (j = TestImage_binary.cols - 1; j >= 0; j--) {
				TestImage_binary.at<uchar>(i, j) = 255;
			}
		}
	}
	else if(direction_vertical ==1){
		for (i = 0; i < move_vertical + 1; i++) {
			for (j = 0; j < TestImage_binary.cols; j++) {
				TestImage_binary.at<uchar>(i, j) = 255;
			}
		}
	}
	if (direction_horizontal == -1) {
		for (i = 0; i < TestImage_binary.rows; i++) {
			for (j = TestImage_binary.cols - 1; j >= TestImage_binary.cols - move_horizontal - 1; j--) {
				TestImage_binary.at<uchar>(i, j) = 255;
			}
		}
	}
	else if (direction_horizontal == 1) {
		for (i = 0; i < TestImage_binary.rows; i++) {
			for (j = 0; j < move_horizontal; j++) {
				TestImage_binary.at<uchar>(i, j) = 255;
			}
		}
	}

	imshow("(modified)", TestImage_binary);
	

	//Enlarge the image if the character is too small
	if (left_pixel + right_pixel > 100) {
		int character_width = width - (right_pixel + left_pixel);
		int character_height = height - (up_pixel + down_pixel);
		int left_margin, right_margin, up_margin, down_margin;
		left_margin = character_width *7  / 11 / 2;
		right_margin = character_width *7 / 11 / 2;
		up_margin = (170 - character_height) / 2;
		down_margin = (170 - character_height) / 2;
		//cout << left_pixel << endl << right_pixel <<endl;
		//cout << character_width << endl << character_height << endl << left_margin << endl << right_margin << endl << up_margin << endl << down_margin << endl;

		//To crop the image
		Mat ROI(TestImage_binary, Rect(left_pixel - left_margin, up_pixel - up_margin, character_width + right_margin + left_margin, character_height + down_margin + up_margin));
		Mat enlarged_image;
		ROI.copyTo(enlarged_image);

		//Enlarge the image by using resize
		resize(enlarged_image, enlarged_image, Size(width, height));
		imshow("Enlarged image", enlarged_image);

		//To get the feature vector of the testing image by using enlarged_image if it is too small
		black = 0, sum1 = 0, sum2 = 0, count = 0;
		for (i = 0; i < enlarged_image.rows; i++) {
			for (j = 0; j < enlarged_image.cols; j++) {
				if ((int)enlarged_image.at<uchar>(i, j) == 0) {
					black++;
				}
				sum2++;
				if (sum2 == p) {
					sum2 = 0;
					Feature_for_test_modified[count] += black;
					black = 0;
					count++;
				}
			}
			sum1++;
			count -= int(width / p);
			if (sum1 == q) {
				count += int(width / p);
				sum1 = 0;
			}
		}

		for (int m = 0; m < size; m++) {
			Feature_for_test_modified[m] = double(Feature_for_test_modified[m] / p / q);
		}
	}
	
	//To get the feature vector of the testing image by using its original binary image if its size is appropriate
	else {
		black = 0, sum1 = 0, sum2 = 0, count = 0;
		for (i = 0; i < TestImage_binary.rows; i++) {
			for (j = 0; j < TestImage_binary.cols; j++) {
				if ((int)TestImage_binary.at<uchar>(i, j) == 0) {
					black++;
				}
				sum2++;
				if (sum2 == p) {
					sum2 = 0;
					Feature_for_test_modified[count] += black;
					black = 0;
					count++;
				}
			}
			sum1++;
			count -= int(width / p);
			if (sum1 == q) {
				count += int(width / p);
				sum1 = 0;
			}
		}

		for (int m = 0; m < size; m++) {
			Feature_for_test_modified[m] = double(Feature_for_test_modified[m] / p / q);
		}
	}
	
	//To compare the distances between the feature vector of the testing image and the feature controids (6 prototypes)
	double eu_distance[6];
	for (i = 0; i < 6; i++) {
		eu_distance[i] = 0;
		for (j = 0; j < size; j++) {
			eu_distance[i] += (double)pow(Feature_for_test_modified[j]-features_centroid[i][j],2);
		}
		eu_distance[i] = (double)pow(eu_distance[i], 0.5);
		cout <<"The distance to character " + labels[i] + "is: " + to_string(eu_distance[i])<<endl;
	}

	int position = distance(eu_distance,min_element(eu_distance, eu_distance + 6)); //Find the smallest distance
	cout << endl << "The character is: " + labels[position] << endl;

	//Test by training characters (To test the algorithm)
	int correct = 0,dett=0,name=0,position;
	double accuracy;
	double distances[6];
	for (i = 0; i < 36; i++) {
		cout << labels[name] + "\t--->\t";
		for (j = 0; j < 6; j++) {
			distances[j] = 0;
			for (k = 0; k < size; k++) {
				distances[j]+= (double)pow(features[i][k] - features_centroid[j][k], 2);
			}
			distances[j] = (double)pow(distances[j], 0.5);
		}
		position = distance(distances, min_element(distances, distances + 6));
		cout << labels[position] + "\n";
		if (name == position) {
			correct++;
		}
		dett++;
		if (dett == 6) {
			dett = 0;
			name++;
		}
	}
	accuracy = (double)correct *100 / 36;
	cout << endl << "The accuracy is: " + to_string(accuracy) + "%" << endl;

	*/

	//Test by new characters with 5 different fonts
	det = 0, shift = 0;
	Mat image_new[30];
	Mat image_binary_new[30];
	vector<vector<double>>features_new(30);
	array<string, 6>names{ "shi", "jie", "yi" ,"fan","ke","ai" };  //For reading files

	//Assign size to the feature vector
	for (i = 0; i < 30; i++) {
		features_new[i].resize(size);
	}

	//Get the feature vector of testing images
	for (k = 0; k < 30; k++) {
		black = 0, sum1 = 0, sum2 = 0, count = 0;
		image_new[k] = imread(names[det] + to_string(shift+1)+".png", 1);  //Read images
		cvtColor(image_new[k], image_new[k], COLOR_RGB2GRAY);  //Convert to grayscale
		threshold(image_new[k], image_binary_new[k], 100, 255, THRESH_BINARY);  //Convert to binary images
		resize(image_binary_new[k], image_binary_new[k], Size(width, height));  //Resize the images for easy manipulation
		//imshow(names[det] + to_string(shift + 1), image_binary_new[k]);
		
		//Count the black pixels
		for (i = 0; i < image_binary_new[k].rows; i++) {
			for (j = 0; j < image_binary_new[k].cols; j++) {
				if ((int)image_binary_new[k].at<uchar>(i, j) == 0) {
					black++;
				}
				sum2++;
				if (sum2 == p) {
					sum2 = 0;
					features_new[k][count] += black;  //Store in the kth feature vector
					black = 0;
					count++;
					//cout << count <<endl;
				}
			}
			sum1++;  //Change block in columns
			count -= int(width / p);
			if (sum1 == q) {  //Change block in rows
				count += int(width / p);
				sum1 = 0;
			}
		}

		//Calculate the ratio of black pixels in one block
		for (int m = 0; m < size; m++) {
			features_new[k][m] = double(features_new[k][m] / p / q);
		}
		//cout << det;
		shift++;  //Change characters
		if (shift == 5) {  
			det++;
			shift = 0;
		}
		
	}
	
	int correct = 0, dett = 0, name = 0, position;
	double accuracy;
	double distances[6];  //Store the distances to 6 feature centroid

	//Get feature vectors of testing images
	for (i = 0; i < 30; i++) {  //Iterate for 30 testing images
		cout << labels[name] + "\t--->\t";   //Print the correct labels
		for (j = 0; j < 6; j++) {  //Iterate for 6 categories
			distances[j] = 0;  //Initialize the distance array
			for (k = 0; k < size; k++) {  //Iterate for all the elements in the feature vector
				distances[j] += (double)pow(features_new[i][k] - features_centroid[j][k], 2);  
			}
			distances[j] = (double)pow(distances[j], 0.5);  //Calculate the distance from feature vector of testing image and 6 centroid feature vectors
		}
		position = distance(distances, min_element(distances, distances + 6));  //Try to find the minimal distance (the potential answer)
		cout << labels[position] + "\n";  //Print the predicted labels
		if (name == position) {  //Check if the predicted label matches the real one
			correct++;
		}
		dett++;
		if (dett == 5) {  //For iterately reading the files
			dett = 0;
			name++;
		}
	}

	//Calculate the recognition accuracy (using the new fonts of characters)
	accuracy = (double)correct * 100 / 30; 
	cout << endl << "The accuracy is: " + to_string(accuracy) + "%" << endl;

	waitKey(0);
	return 0;
}
