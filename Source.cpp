#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <omp.h>

using namespace cv;
using namespace std;

//PacMan map in matrix form
const bool mazeMatrix[21][21] = {
{0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0},
{0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0},
{0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0},
{0,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,0},
{0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0},
{0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0},
{0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0},
{0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0},
{0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0},
{0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0},
{0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0},
{1,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1},
{0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0},
{0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0},
{0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0},
{0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0},
{0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0},
{0,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,0},
{0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0},
{1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1},
{0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0}
};

// Food map
bool foodMatrix[21][21] = {
{0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0},
{0,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,0},
{0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0},
{0,1,0,1,1,1,0,1,0,1,1,1,1,1,0,1,1,1,1,1,0},
{0,1,0,0,0,0,0,1,0,1,0,0,0,1,0,1,0,1,0,0,0},
{0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0},
{0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0},
{0,1,1,1,0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0},
{0,1,0,1,0,1,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0},
{0,1,0,1,1,1,0,1,1,1,0,1,0,1,0,1,0,1,1,1,0},
{0,1,0,0,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,1,0},
{1,1,1,1,1,1,0,1,1,1,0,1,0,1,0,1,1,1,1,1,1},
{0,0,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0},
{0,1,1,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,1,1,0},
{0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0},
{0,1,1,1,0,1,0,1,1,1,1,1,1,1,1,1,0,1,1,1,0},
{0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0,1,0,1,0},
{0,1,0,1,0,1,1,1,1,1,1,1,0,1,1,1,0,1,0,1,0},
{0,1,0,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0},
{1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,1,1,1,0,1,1},
{0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,0}
};

//Inital pacman position
int px = 20;
int py = 20;
int counter = 0;
cv::Mat food = cv::imread("C:/Users/Pera/Desktop/Project1/images/food.png");

void eat(cv::Mat& map,int px,int py)
{
	if (foodMatrix[py / 20][px / 20]) {
		counter++;
		foodMatrix[py / 20][px / 20] = 0;
		cv::Rect roi(cv::Point(px, py), food.size());
		cv::rectangle(map, roi, cv::Scalar(255, 255, 255), -1);
	}

}
void movement(int key,cv::Mat& map)
{
	switch (key)
	{
		case 'w': 
				if (py <= 0)
					py = 400;
				else if(mazeMatrix[(py - 20) / 20][px / 20] == 1)
					py = py - 20;
			eat(map, px, py);
			cout << "U pressed w and new location is : " << px/20 << "," << py/20 << endl;
			break;

		case 's':
			if (py == 400)
				py = 0;
			else if (mazeMatrix[(py + 20) / 20][px / 20] == 1)
				py = py + 20;

			eat(map, px, py);
			cout << "U pressed s and new location is : " << px/20 << "," << py/20 << endl;
			break;

		case 'a':
			if (px == 0)
				px = 400;
			else if (mazeMatrix[py / 20][(px - 20) / 20] == 1)
				px = px - 20;

			eat(map, px, py);
			cout << "U pressed a and new location is : " << px/20 << "," << py/20 << endl;
			break;

		case 'd':
			if (px == 400)
				px = 0;
			else if (mazeMatrix[py/20][(px + 20) / 20] == 1)
				px = px + 20;
			
			eat(map, px, py);
			cout << "U pressed d and new location is : " << px/20 << "," << py/20 << endl;
			break;
	}

}

void drawPacman(cv::Mat& image) 
{
	/*
	cv::Mat pacman = cv::imread("C:/Users/Pera/Desktop/Project1/images/pacman.png");
	if (pacman.empty()) {
		cout << "Error in loading images." << endl;
		return;
	}

	cv::Size newSize(20, 20);
	cv::Mat Pac20;
	cv::resize(pacman, Pac20, newSize);

	cv::Rect roi(cv::Point(px, py), Pac20.size());
	cv::Mat desinationROI = image(roi);
	cv::Mat pacmanMask = Pac20(cv::Rect(0, 0, Pac20.cols, Pac20.rows));
	Pac20.copyTo(desinationROI, pacmanMask);
	*/
	

	cv::Scalar color(255, 255, 0); // Yellow

	// Pacman center coordinates
	cv::Point center(px + 10, py + 10); // Adjusted to center Pacman in the grid cell

	// Pacman radius
	int radius = 10; // Adjusted to fit Pacman within the grid cell

	// Draw Pacman
	cv::circle(image, center, radius, color, -1);
}

void viewPoints(cv::Mat& image)
{
	string points = "Points collected : " + to_string(counter);
	putText(image, points, Point(20, 20), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
	imshow("Points", image);

	if(counter == 210)
	{
		string win = "You collected all points !";
		putText(image, win, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
		imshow("Points", image);
	}
}

int main()
{
	//cv::Mat map = cv::imread("C:/Users/Pera/Desktop/Project1/images/map.png");
	

	if (/*map.empty()||*/ food.empty()) {
		cout << "Error in loading images." << endl;
		return -1;
	}

	// Create a map accordingly to a maze matrix
	const int scale = 20;
	cv::Mat map = cv::Mat::zeros(21 * scale, 21 * scale, CV_8UC3);
	for (int i = 0;i < 21; i++)
	{
		for (int j = 0; j < 21; j++) {
			cv::Scalar color = mazeMatrix[i][j] ? cv::Scalar(255, 255, 255) : cv::Scalar(0, 0, 0);
			cv::rectangle(map, cv::Point(j * scale, i * scale), cv::Point((j + 1) * scale, (i + 1) * scale), color, -1);
		}
	}

	// Fill the map with food
	for (int i = 0; i < 21; i++) {
		for (int j = 0; j < 21; j++) {
			if (mazeMatrix[i][j] == 1)
			{
				int fx = j * (map.cols / 21);
				int fy = i * (map.rows / 21);
				cv::Rect roi2(cv::Point(fx, fy), food.size());
				cv::Mat destinantionROI2 = map(roi2);
				cv::Mat foodMask = food(cv::Rect(0, 0, food.cols, food.rows));
				food.copyTo(destinantionROI2, foodMask);
			}
		}
	}

	int zeroind = 0;
	for (int i = 0; i < 21; i++) {
		for (int j = 0; j < 21; j++) {
			if (mazeMatrix[i][j] == 0)
				zeroind++;
		}
	}
	cout << "NUmer of zeros is : " << zeroind << endl;
	
	while (true)
	{
		cv::Mat display = map.clone();
		cv::Mat pointsCounter = Mat::zeros(100, 400, CV_8UC3);
		drawPacman(display);
		viewPoints(pointsCounter);
		cv::imshow("Pacman Game", display);
		char key = cv::waitKey(100);
		if (key == 27)
			break;
		else
			movement(key,map);
	}

	// Testing pacman positioning
	/*
	cv::Rect roi(cv::Point(px,py),pacman.size());
	cv::Mat desinationROI = map(roi);
	cv::Mat pacmanMask = pacman(cv::Rect(0, 0, pacman.cols, pacman.rows));
	pacman.copyTo(desinationROI, pacmanMask);
	*/
	//cv::imshow("Map with food and pacman", map);
	//cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}