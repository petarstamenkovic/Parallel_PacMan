#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <omp.h>
#include <iostream>

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

// Inital ghost one position
int g1x = 20;
int g1y = 220;

// Initial ghost two position
int g2x = 320;
int g2y = 300;

// Collected food counter
int counter = 0;

cv::Mat food = cv::imread("C:/Users/Pera/Desktop/Project1/images/food.png");

void eat(cv::Mat& map, int px, int py)
{
	if (foodMatrix[py / 20][px / 20]) {
		counter++;
		foodMatrix[py / 20][px / 20] = 0;
		cv::Rect roi(cv::Point(px, py), food.size());
		cv::rectangle(map, roi, cv::Scalar(255, 255, 255), -1);
	}

}
void movement(int key, cv::Mat& map)
{
	switch (key)
	{
	case 'w':
	case 'W':  
		if (py <= 0)
			py = 400;
		else if (mazeMatrix[(py - 20) / 20][px / 20] == 1)
			py = py - 20;
		eat(map, px, py);
		break;

	case 's':
	case 'S':
		if (py == 400)
			py = 0;
		else if (mazeMatrix[(py + 20) / 20][px / 20] == 1)
			py = py + 20;

		eat(map, px, py);
		break;

	case 'a':
	case 'A':
		if (px == 0)
			px = 400;
		else if (mazeMatrix[py / 20][(px - 20) / 20] == 1)
			px = px - 20;

		eat(map, px, py);
		break;

	case 'd':
	case 'D':
		if (px == 400)
			px = 0;
		else if (mazeMatrix[py / 20][(px + 20) / 20] == 1)
			px = px + 20;

		eat(map, px, py);
		break;
	}
}

void ghostMovement1()
{
	int direction = rand() % 4;
	switch (direction)
	{
		
	case 0: // Up 
		if (g1y <= 0)
			g1y = 400;
		else if (mazeMatrix[(g1y - 20) / 20][g1x / 20] == 1)
			g1y = g1y - 20;
		break;

		 
	case 1: // Down
		if (g1y == 400)
			g1y = 0;
		else if (mazeMatrix[(g1y + 20) / 20][g1x / 20] == 1)
			g1y = g1y + 20;
		break;

		
	case 2: // Left
		if (g1x == 0)
			g1x = 400;
		else if (mazeMatrix[g1y / 20][(g1x - 20) / 20] == 1)
			g1x = g1x - 20;
		break;

		
	case 3: // Right
		if (g1x == 400)
			g1x = 0;
		else if (mazeMatrix[g1y / 20][(g1x + 20) / 20] == 1)
			g1x = g1x + 20;
		break;
	}
}

void drawGhost1(cv::Mat& image)
{

	cv::Mat ghost1 = cv::imread("C:/Users/Pera/Desktop/Project1/images/red_ghost.png");
	if (ghost1.empty()) {
		cout << "Error in loading images." << endl;
		return;
	}

	cv::Size newSize(20, 20);
	cv::Mat ghost1_20;
	cv::resize(ghost1, ghost1_20, newSize);

	cv::Rect roi(cv::Point(g1x, g1y), ghost1_20.size());
	cv::Mat desinationROI2 = image(roi);
	cv::Mat ghost1_20_Mask = ghost1_20(cv::Rect(0, 0, ghost1_20.cols, ghost1_20.rows));
	ghost1_20.copyTo(desinationROI2, ghost1_20_Mask);

	/*
	cv::Scalar color(0, 0, 255);
	cv::Point center(g1x + 10, g1y + 10);
	int radius = 10;
	cv::circle(image, center, radius, color, -1);
	*/
}

void ghostMovement2()
{
	int direction = rand() % 4;
	switch (direction)
	{
			
	case 0: // Up 
		if (g2y <= 0)
			g2y = 400;
		else if (mazeMatrix[(g2y - 20) / 20][g2x / 20] == 1)
			g2y = g2y - 20;
		break;

		
	case 1: // Down 
		if (g2y == 400)
			g2y = 0;
		else if (mazeMatrix[(g2y + 20) / 20][g2x / 20] == 1)
			g2y = g2y + 20;
		break;

		
	case 2: // Left
		if (g2x == 0)
			g2x = 400;
		else if (mazeMatrix[g2y / 20][(g2x - 20) / 20] == 1)
			g2x = g2x - 20;
		break;

		
	case 3: // Right
		if (g2x == 400)
			g2x = 0;
		else if (mazeMatrix[g2y / 20][(g2x + 20) / 20] == 1)
			g2x = g2x + 20;
		break;
	}
}

void drawGhost2(cv::Mat& image)
{
	cv::Mat ghost2 = cv::imread("C:/Users/Pera/Desktop/Project1/images/blue_ghost.png");
	if (ghost2.empty()) {
		cout << "Error in loading images." << endl;
		return;
	}

	cv::Size newSize(20, 20);
	cv::Mat ghost2_20;
	cv::resize(ghost2, ghost2_20, newSize);

	cv::Rect roi(cv::Point(g2x, g2y), ghost2_20.size());
	cv::Mat desinationROI3 = image(roi);
	cv::Mat ghost2_20_Mask = ghost2_20(cv::Rect(0, 0, ghost2_20.cols, ghost2_20.rows));
	ghost2_20.copyTo(desinationROI3, ghost2_20_Mask);

	/*
	cv::Scalar color(0, 0, 255);
	cv::Point center(g2x + 10, g2y + 10);
	int radius = 10;
	cv::circle(image, center, radius, color, -1);
	*/
}

void drawPacman(cv::Mat& image)
{
	// If for some reason this gives opencv roi errors, use the code below (Yellow circle instead of actual image.)
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

	/*
	cv::Scalar color(0, 255, 255); // Yellow
	cv::Point center(px + 10, py + 10);
	int radius = 10;
	cv::circle(image, center, radius, color, -1);
	*/
}

void victory(cv::Mat& image)
{
	string win = "Victory!";
	putText(image, win, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
	imshow("PacMan Game", image);
}

void defeat(cv::Mat& image)
{
	string defeat = "Defeat!";
	string points_won = "Points collected : " + to_string(counter);
	putText(image, defeat, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
	putText(image, points_won, Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
	imshow("PacMan game", image);
}
int main(int argc, char* argv[])
{
    // Number of threads
	int tc = strtol(argv[1], NULL, 10);

	srand(time(NULL));
	if (food.empty()) {
		cout << "Error in loading images." << endl;
		return -1;
	}

	// Create a map
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
	//////////////// parallel code /////////////////
	while (true)
	{
		char key = cv::waitKey(100);
		// Key 27 is an escape key
		if (key == 27)
			break;
		else
		{
			cv::Mat display = map.clone();
			cv::Mat finalWindow = Mat::zeros(100, 400, CV_8UC3);
			#pragma omp parallel num_threads(tc)
			{
				int trank = omp_get_thread_num();
				switch (trank)
				{
				case 0: // Thread 0 controls the changes in the map and drawing out charachers
					drawPacman(display);
					drawGhost1(display);
					drawGhost2(display);
					cv::imshow("Pacman Game", display);
					//cout << "Hi im thread 0! " << endl;
					break;

				case 1:// Pacmans movement
					movement(key, map);
					//cout << "Hi im thread 1! " << endl;
					break;

				case 2: // Ghost 1 movement
					ghostMovement1();
					//cout << "Hi im thread 2! " << endl;
					break;

				case 3: // Ghost 2 movement
					ghostMovement2();
					//cout << "Hi im thread 3! " << endl;
					break;

				}
			}

			// This is not parallel becaues its the endgame mechanism
			if (counter == 210)
			{
				victory(finalWindow);
				break;
			}
			// Check if pacman and ghost collided
			if ((px == g1x && py == g1y) || (px == g2x && py == g2y))
			{
				defeat(finalWindow);
				break;
			}
		}

	}
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

