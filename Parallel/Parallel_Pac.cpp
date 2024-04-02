#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
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

// Change to your path directory
cv::Mat food = cv::imread("C:/Users/Pera/Desktop/Project1/images/food.png");

// Function that handles food removal from the food map
void eat(cv::Mat& map, int px, int py)
{
	if (foodMatrix[py / 20][px / 20]) 
	{
		counter++;
		foodMatrix[py / 20][px / 20] = 0;
		cv::Rect roi(cv::Point(px, py), food.size());
		cv::rectangle(map, roi, cv::Scalar(255, 255, 255), -1);
	}

}

// Function that moves the pacman depending on a key (for some reason arrows dont seem to work for my machine)
void movement(int key, cv::Mat& map)
{
	switch (key)		// Arrow keys on my keyboard dont give any feedback for cv::waitKey(100) so I use only w a s d
						// If you want you can add arrow keys on your machine - its usually something around 70-80
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

// Function that controls the movement of the first ghost
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

// Function that draws the first ghost
void drawGhost1(cv::Mat& image)
{
	// Change to your path directory
	cv::Mat ghost1 = cv::imread("C:/Users/Pera/Desktop/Project1/images/red_ghost.png");
	if (ghost1.empty()) {
		cout << "Error in loading images." << endl;
		return;
	}

	// Resize to fit the map image
	cv::Size newSize(20, 20);
	cv::Mat ghost1_20;
	cv::resize(ghost1, ghost1_20, newSize);

	cv::Rect roi(cv::Point(g1x, g1y), ghost1_20.size());
	cv::Mat desinationROI2 = image(roi);
	cv::Mat ghost1_20_Mask = ghost1_20(cv::Rect(0, 0, ghost1_20.cols, ghost1_20.rows));
	ghost1_20.copyTo(desinationROI2, ghost1_20_Mask);

	// If picture loading does not work for some reason uncomment this and comment upper code
	/*
	cv::Scalar color(0, 0, 255);
	cv::Point center(g1x + 10, g1y + 10);
	int radius = 10;
	cv::circle(image, center, radius, color, -1);
	*/
}

// Function that controls the movement of ghost 2
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

// Function that draws the second ghost
void drawGhost2(cv::Mat& image)
{
	// Change to your path directory
	cv::Mat ghost2 = cv::imread("C:/Users/Pera/Desktop/Project1/images/blue_ghost.png");
	if (ghost2.empty()) {
		cout << "Error in loading images." << endl;
		return;
	}

	// Resize to fit the map image
	cv::Size newSize(20, 20);
	cv::Mat ghost2_20;
	cv::resize(ghost2, ghost2_20, newSize);

	cv::Rect roi(cv::Point(g2x, g2y), ghost2_20.size());
	cv::Mat desinationROI3 = image(roi);
	cv::Mat ghost2_20_Mask = ghost2_20(cv::Rect(0, 0, ghost2_20.cols, ghost2_20.rows));
	ghost2_20.copyTo(desinationROI3, ghost2_20_Mask);

	// If picture loading does not work for some reason uncomment this and comment the upper code
	/*
	cv::Scalar color(0, 0, 255);
	cv::Point center(g2x + 10, g2y + 10);
	int radius = 10;
	cv::circle(image, center, radius, color, -1);
	*/
}

// Function that draws the pacman
void drawPacman(cv::Mat& image)
{
	// If for some reason this gives opencv roi errors, use the code below (Yellow circle instead of actual image.)
	// Change to your path directory
	cv::Mat pacman = cv::imread("C:/Users/Pera/Desktop/Project1/images/pacman.png");
	if (pacman.empty()) {
		cout << "Error in loading images." << endl;
		return;
	}

	// Resize to fit the map image
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

// Function that draws the winning window
void victory(cv::Mat& image)
{
	string win = "Victory!";
	putText(image, win, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
	imshow("Pacman", image);
}

// Function that draws the losing window
void defeat(cv::Mat& image)
{
	string defeat = "Defeat!";
	string points_won = "Points collected : " + to_string(counter);
	putText(image, defeat, Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
	putText(image, points_won, Point(20, 60), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
	imshow("Pacman", image);
}

// MAIN CODE
int main(int argc, char* argv[])
{
	// Here check the ASCI values for arrows if u want, breakpoint it at 331
	/*char key_press;
	while (true)
	{
		char key = cv::waitKey(100);
		//if(key_press != -1)
			cout << "You pressed the key with code : " << key << endl;
	}*/
	

    // Number of threads
	int tc = strtol(argv[1], NULL, 10);

	// Random seed for ghost movement
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

	// True - Never exist the loop unless winning mechanism is achieved or esc key is pressed
	while (true)
	{
		char key = cv::waitKey(100);
		//cout << "You pressed the key with code : " << (int)key << endl;
		// Key 27 is an escape key
		if (key == 27)
			break;
		else
		{
			cv::Mat display = map.clone();		// Clone the map and do changes on that one so that icons dont stack up
			cv::Mat finalWindow = Mat::zeros(100, 400, CV_8UC3);
			
			#pragma omp parallel num_threads(tc)
			{
				int trank = omp_get_thread_num();
				// Depends on a thread rank do a specific block of code -- Cout statements are for debugging purposes
				switch (trank)
				{
				case 0: // Thread 0 controls the changes in the map and drawing out charachers
					drawPacman(display);
					drawGhost1(display);
					drawGhost2(display);
					cv::imshow("Pacman", display);
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

			// This is not parallel becaues it's the endgame mechanism
			if (counter == 210)
			{
				victory(finalWindow);
				break;
			}

			// Check if pacman and ghost collided - This means you lost
			if ((px == g1x && py == g1y) || (px == g2x && py == g2y))
			{
				defeat(finalWindow);
				break;
			}
		}

	}
	
	// After exiting the main while loop (endgame), wait for any key press 
	// so the game does not immidiatelly close
	//cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

