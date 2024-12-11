#define STB_IMAGE_IMPLEMENTATION
#include "link/stb_image.h"

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <algorithm>
#include <random>

#include <fstream>  
#include <string>

//#include <windows.h>
#include <pthread.h>



#define PI 3.1415926
#define P2 PI/2
#define P3 3*PI/2
//#define DEG 0.0174533

#define WINDOW_WIDTH 1024
#define WINDOW_HEIGHT 512

// Global variables to hold dynamic screen resolution
int screenWidth = WINDOW_WIDTH;
int screenHeight = WINDOW_HEIGHT;

#define FOV (PI / 3)  // 60 degrees FOV

#define NUM_OF_RAYS 120 //30 //60 //960 //480 //240 //120
#define RAY_ANGLE_INC (FOV / NUM_OF_RAYS)
#define MAP_SIZE 225//should be a multiple of MAP_ARRAY
#define MAP_ARRAY 15//the size of our map in cells, maps always square
#define MAP_CELL_SIZE MAP_SIZE/MAP_ARRAY
#define PLAYER_SIZE    MAP_CELL_SIZE/3
#define VIEW_STRIP    WINDOW_WIDTH/NUM_OF_RAYS
#define TURNING_ANGLE 0.07
#define SPEED    PLAYER_SIZE/7

#define MOUSE_SENSITIVITY 0.0005

#define NUM_WALL_TYPES 2 
GLuint wallTextures[NUM_WALL_TYPES];




const int NUM_ACTIONS = 4;  // Forward, Backward, Left, Right
float qTable[MAP_ARRAY * MAP_ARRAY][NUM_ACTIONS];  // Q-table: State -> Action

float alpha = std::max(0.01f, alpha * 0.995f);  // Gradually decay learning rate
float epsilon = std::max(0.1f, epsilon * 0.99f); // Slower epsilon decay for more exploration
float gammaValue = 0.9f;  // Discount factor

// Initialize the Q-table with zeros
void initializeQTable() {
    for (int i = 0; i < MAP_ARRAY * MAP_ARRAY; ++i) {
        for (int j = 0; j < NUM_ACTIONS; ++j) {
            qTable[i][j] = 0.0f; // Initialize all Q-values to zero
        }
    }
}



bool isTraining = false;
int currentEpisode = 0;





//puts them in the middle of cell 1,1 facing right

float playerX = MAP_CELL_SIZE * 1.5;
float playerY = MAP_CELL_SIZE * 0.5;

float playerAngle = 1.5;

int keyStates[256] = { 0 };


std::mt19937 g(static_cast<unsigned int>(std::time(nullptr)));


enum GameState {
    MENU,
    PAUSE_MENU,
    GAME
};
GameState currentState = MENU;

unsigned int score;



int map[MAP_SIZE] = { 0 };



void updateMovementWithQLearning() {
    // Get the current state (player's position)
    int currentState = ((int)(playerY / MAP_CELL_SIZE) * MAP_ARRAY) + ((int)(playerX / MAP_CELL_SIZE));

    // Define goal position in map coordinates
    float goalX = (MAP_ARRAY - 1) * MAP_CELL_SIZE;
    float goalY = (MAP_ARRAY - 1) * MAP_CELL_SIZE;

    // Choose action based on epsilon-greedy strategy
    int action;
    if ((float)rand() / RAND_MAX < epsilon) {
        // Exploration: Random action
        action = rand() % NUM_ACTIONS;
    }
    else {
        // Exploitation: Choose best action based on Q-table
        action = 0;
        for (int i = 1; i < NUM_ACTIONS; ++i) {
            if (qTable[currentState][i] > qTable[currentState][action]) {
                action = i;
            }
        }
    }

    // Perform the action (move the player)
    float nextX = playerX;
    float nextY = playerY;
    float reward = -1.0f;  // Default movement penalty

    switch (action) {
    case 0: // Move forward
        nextX += cos(playerAngle) * SPEED;
        nextY += sin(playerAngle) * SPEED;
        break;
    case 1: // Move backward
        nextX -= cos(playerAngle) * SPEED;
        nextY -= sin(playerAngle) * SPEED;
        break;
    case 2: // Turn left
        playerAngle -= TURNING_ANGLE;
        break;
    case 3: // Turn right
        playerAngle += TURNING_ANGLE;
        break;
    }

    if (nextX < 0 || nextX >= MAP_SIZE || nextY < 0 || nextY >= MAP_SIZE) {
        reward = -20.0f;  // Larger penalty for attempting to move out of bounds
        nextX = playerX;  // Revert position to prevent movement
        nextY = playerY;
    }

    // Calculate next position index in the map array
    int nextState = ((int)nextY / ((int)MAP_CELL_SIZE)) * MAP_ARRAY + ((int)nextX / ((int)MAP_CELL_SIZE));

    // Check for wall collision and path-following
    if (map[nextState] != 0) {  // Collision with wall
        reward = -10.0f;  // Larger penalty for hitting a wall
        nextX = playerX;  // Reset to current position if wall hit
        nextY = playerY;
    }
    else {
        reward = 0.5f;  // Small reward for staying on the path (zero cell)

        // Calculate distance to the goal
        float distanceToGoal = sqrt(pow(nextX - goalX, 2) + pow(nextY - goalY, 2));

        // Give higher reward as the agent gets closer to the goal
        reward += (1.0f / (distanceToGoal + 1.0f)) * 10.0f;  // Inverse of distance

        // Special large reward if reaching the exact goal position
        if ((int)(nextX / MAP_CELL_SIZE) == MAP_ARRAY - 1 && (int)(nextY / MAP_CELL_SIZE) == MAP_ARRAY - 1) {
            reward = 200.0f;  // Large reward for reaching the goal
        }
    }

    // Update Q-value using the Q-learning formula
    float maxNextQ = *std::max_element(qTable[nextState], qTable[nextState] + NUM_ACTIONS);
    qTable[currentState][action] = qTable[currentState][action] + alpha * (reward + gammaValue * maxNextQ - qTable[currentState][action]);

    // Update player position if no wall was hit
    playerX = nextX;
    playerY = nextY;

    // Trigger re-render
    glutPostRedisplay();
}







void loadState(const std::string& filename) {
    std::ifstream inFile(filename); // Open the file for reading
    if (inFile.is_open()) {
        std::string line;
        while (std::getline(inFile, line)) {
            if (line.find("score=") == 0) { // Check if the line starts with "score="
                score = std::stoi(line.substr(6)); // Extract the number after "score="
            }
        }
        inFile.close();  // Close the file
    }
    else {
        std::cerr << "Unable to open file for reading: " << filename << std::endl;
        score = 0; // Default score if file doesn't exist or is unreadable
    }
}



void saveState(const std::string& filename) {
    std::ofstream outFile(filename); // Open the file for writing
    if (outFile.is_open()) {
        outFile << "score=" << score << std::endl; // Write the score in the desired format
        outFile.close();  // Close the file
    }
    else {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
    }
}


void initializeMapWithWalls() {
    for (int i = 0; i < MAP_ARRAY; i++) {
        for (int j = 0; j < MAP_ARRAY; j++) {
            map[i * MAP_ARRAY + j] = 1; // Set all cells to walls initially
        }
    }
}


// Recursive function for depth-first search-based maze generation
void carveMaze(int x, int y, std::mt19937& g) {
    int directions[4][2] = { {0, 1}, {1, 0}, {0, -1}, {-1, 0} };

    // Random device and generator for shuffling directions


    std::shuffle(std::begin(directions), std::end(directions), g); // Add the generator here

    for (int i = 0; i < 4; i++) {
        int nx = x + directions[i][0] * 2;
        int ny = y + directions[i][1] * 2;

        // Check boundaries and ensure the next cell is a wall (unvisited)
        if (nx >= 0 && nx < MAP_ARRAY && ny >= 0 && ny < MAP_ARRAY && map[ny * MAP_ARRAY + nx] == 1) {
            // Carve path to the next cell
            map[ny * MAP_ARRAY + nx] = 0; // Mark the next cell as a path
            map[(y + directions[i][1]) * MAP_ARRAY + (x + directions[i][0])] = 0; // Mark the wall between cells as a path
            carveMaze(nx, ny, g);  // Recurse
        }
    }
}



// Main function to generate the maze
void generateMaze() {
    initializeMapWithWalls(); // Initialize map with walls

    // Set the player's starting position in the map to be a path (0)
    int startX = 1; // Starting cell X coordinate
    int startY = 1; // Starting cell Y coordinate

    map[0 * MAP_ARRAY] = 0;
    map[0 * MAP_ARRAY + 1] = 0;

    // Carve the maze from the starting position
    carveMaze(startX, startY, g);

    // Ensure the last position of the map is a path (0) with enough space
    int endX = MAP_ARRAY - 1; // Last column
    int endY = MAP_ARRAY - 1; // Last row

    // Set the ending cells (14,14) and (13,14) to free
    map[14 * MAP_ARRAY + 14] = 0; // Set (14,14) to free
    map[13 * MAP_ARRAY + 14] = 0; // Set (13,14) to free

    // Ensure the last position is a path (0)
    // Check if end position is valid
    if (map[endY * MAP_ARRAY + endX] == 1) {
        map[endY * MAP_ARRAY + endX] = 0; // Mark end cell as path (0)
    }
}




void resetMazeAndPlayer() {
    generateMaze();  // Generate a new maze
    playerX = MAP_CELL_SIZE * 1.5;  // Reset player X position
    playerY = MAP_CELL_SIZE * 0.5;  // Reset player Y position
    playerAngle = 1.5;              // Reset player angle if needed
}


void trainAgent(int episodes) {
    for (int episode = 0; episode < episodes; ++episode) {
        //resetMazeAndPlayer();  // Reset maze and player at the start of each episode
        while (true) {
            updateMovementWithQLearning();
            if (playerX > MAP_CELL_SIZE * 14 && playerY > MAP_CELL_SIZE * 14) {
                break;  // Goal reached
            }
        }
    }
}

void* trainAgentThread(void* param) {
    trainAgent(1000);  // Train the agent for 100 episodes
    isTraining = false;  // Mark training as complete
    return NULL;
}




struct Ray {
    float distance;
    float angle;
    bool vertical;
    int wallType;
};

Ray rays[NUM_OF_RAYS];


void drawMinimap() {
    // Set the map in the bottom right
    float translateY = WINDOW_HEIGHT - MAP_SIZE;
    float visibilityRadius = 1000.0f; // Adjust this as needed for visibility range

    // Draw the map border
    glColor3f(0, 0, 0);
    glBegin(GL_QUADS);
    glVertex2f(0, translateY);
    glVertex2f(MAP_SIZE, translateY);
    glVertex2f(MAP_SIZE, translateY + MAP_SIZE);
    glVertex2f(0, translateY + MAP_SIZE);
    glEnd();

    // Loop through map
    for (int i = 0; i < MAP_ARRAY; i++) {
        for (int j = 0; j < MAP_ARRAY; j++) {

            // Calculate the center of each cell
            float cellCenterX = MAP_CELL_SIZE * j + MAP_CELL_SIZE / 2;
            float cellCenterY = translateY + MAP_CELL_SIZE * i + MAP_CELL_SIZE / 2;

            // Calculate distance from player to cell
            float dx = cellCenterX - playerX;
            float dy = cellCenterY - (translateY + playerY);
            float distance = sqrt(dx * dx + dy * dy);

            // Only draw walls within visibilityRadius
            if (distance <= visibilityRadius) {
                // Pick color based on wall type
                switch (map[i * MAP_ARRAY + j]) {
                case 0:
                    glColor3f(0, 0, 0);  // Empty space
                    break;
                case 1:
                    glColor3f(0.1, 0.1, 0.1);  // Wall
                    break;
                }

                // Draw cell
                glBegin(GL_QUADS);
                glVertex2f(MAP_CELL_SIZE * j + 1, translateY + MAP_CELL_SIZE * i + 1);
                glVertex2f(MAP_CELL_SIZE * j + MAP_CELL_SIZE - 1, translateY + MAP_CELL_SIZE * i + 1);
                glVertex2f(MAP_CELL_SIZE * j + MAP_CELL_SIZE - 1, translateY + MAP_CELL_SIZE * i + MAP_CELL_SIZE - 1);
                glVertex2f(MAP_CELL_SIZE * j + 1, translateY + MAP_CELL_SIZE * i + MAP_CELL_SIZE - 1);
                glEnd();
            }
        }
    }

    // Draw player
    glPointSize(PLAYER_SIZE);
    glColor3f(1, 0, 0);  // Red color for player
    glBegin(GL_POINTS);
    glVertex2f(playerX, translateY + playerY);
    glEnd();
}




GLuint loadTexture(const char* filename) {
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    int width, height, nrChannels;
    unsigned char* data = stbi_load(filename, &width, &height, &nrChannels, 0);
    if (data) {
        std::cout << "Loaded texture: " << filename << " (" << width << "x" << height << ")" << std::endl;
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else {
        std::cerr << "Failed to load texture: " << filename << std::endl;
    }
    stbi_image_free(data);

    return texture;
}




void loadWallTextures() {
    wallTextures[0] = loadTexture("data/Wall1.png");  // Texture for wallType 1
    wallTextures[1] = loadTexture("data/Wall2.png");  // Texture for wallType 2
    // Continue loading textures as needed
}





void drawView() {

    //draw the floor
    glColor3f(0, 0, 0);
    glBegin(GL_QUADS);
    glVertex2f(0, WINDOW_HEIGHT / 2);
    glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT / 2);
    glVertex2f(WINDOW_WIDTH, WINDOW_HEIGHT);
    glVertex2f(0, WINDOW_HEIGHT);
    glEnd();

    //draw each part of the wall
    float lineHeight;
    float fixFishEye;
    float intensity;
    for (unsigned int i = 0; i < sizeof(rays) / sizeof(Ray); i++) {

        // fix distortion (fish-eye effect)
        fixFishEye = playerAngle - rays[i].angle;
        if (fixFishEye < 0) {
            fixFishEye += 2 * PI;
        }
        else if (fixFishEye > 2 * PI) {
            fixFishEye -= 2 * PI;
        }
        rays[i].distance = rays[i].distance * cos(fixFishEye);

        intensity = 1.0 / (1.0 + 0.01 * rays[i].distance); // Adjust 0.01 for stronger/weaker effect

        switch (rays[i].wallType) {
        case 1:
            if (rays[i].vertical) {
                glColor3f(0.1 * intensity, 0.1 * intensity, 0.1 * intensity);
            }
            else {
                glColor3f(0.05 * intensity, 0.05 * intensity, 0.05 * intensity);
            }
            break;
        }

        // set line height and draw
        lineHeight = (MAP_CELL_SIZE * WINDOW_HEIGHT) / rays[i].distance;
        glBegin(GL_QUADS);
        glVertex2f(i * VIEW_STRIP, WINDOW_HEIGHT / 2 - lineHeight / 2);
        glVertex2f((i + 1) * VIEW_STRIP, WINDOW_HEIGHT / 2 - lineHeight / 2);
        glVertex2f((i + 1) * VIEW_STRIP, WINDOW_HEIGHT / 2 + lineHeight / 2);
        glVertex2f(i * VIEW_STRIP, WINDOW_HEIGHT / 2 + lineHeight / 2);
        glEnd();
    }
}


//helper function for casting rays
float distance(float ax, float bx, float ay, float by) {
    return sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay));
}



//casts a bunch o rays
void castRays() {
    float rayY, rayX, xOffset, yOffset;
    int mapX, mapY, mapIndex, hWallType, vWallType;
    int depth, maxDepth = MAP_ARRAY; //how many times we loop before giving up

    float rayAngle = playerAngle - FOV / 2;

    if (rayAngle < 0) {//constrain angle
        rayAngle += 2 * PI;
    }
    else if (rayAngle > 2 * PI) {
        rayAngle -= 2 * PI;
    }


    for (int i = 0;i < NUM_OF_RAYS; i++) {
        //HORIZONTAL RAY CHECK
        float hX, hY, hDistance = 9999999999;
        depth = 0;
        float aTan = -1 / tan(rayAngle);

        //initial maths to find the ray cords and offsets
        if (rayAngle == PI || rayAngle == 0) {//looking straight left or right, never gonna collide
            rayX = playerX;
            rayY = playerY;
            depth = 8;
        }
        else {
            if (rayAngle > PI) {//looking up
                rayY = ((int)playerY) / ((int)MAP_CELL_SIZE) * ((int)MAP_CELL_SIZE) - 0.0001;
                yOffset = -MAP_CELL_SIZE;
            }
            else {//looking down
                rayY = (((int)playerY) / ((int)MAP_CELL_SIZE) * ((int)MAP_CELL_SIZE)) + MAP_CELL_SIZE;
                yOffset = MAP_CELL_SIZE;
            }
            rayX = (playerY - rayY) * aTan + playerX;
            xOffset = -yOffset * aTan;
        }

        //figure out when we hit a wall
        while (depth < maxDepth) {
            mapX = ((int)rayX) / ((int)MAP_CELL_SIZE);
            mapY = rayY / ((int)MAP_CELL_SIZE);
            mapIndex = mapY * MAP_ARRAY + mapX;
            if (mapIndex >= 0 && mapIndex < MAP_ARRAY * MAP_ARRAY && map[mapIndex] != 0) {//hit a wall
                depth = maxDepth;
                hWallType = map[mapIndex];
                hX = rayX;
                hY = rayY;
                hDistance = distance(playerX, hX, playerY, hY);
            }
            else {
                rayX += xOffset;
                rayY += yOffset;
                depth++;
            }
        }


        //VERTICAL RAY CHECK
        //basically identidentical to horizontal check
        float vX, vY, vDistance = 9999999999;
        depth = 0;
        float nTan = -tan(rayAngle);


        //initial maths to find the ray cords and offsets
        if (rayAngle == P2 || rayAngle == P3) {//looking straight up or down, never gonna collide
            rayX = playerX;
            rayY = playerY;
            depth = 8;
        }
        else {
            if (rayAngle > P2 && rayAngle < P3) {//looking up
                rayX = ((int)playerX) / ((int)MAP_CELL_SIZE) * ((int)MAP_CELL_SIZE) - 0.0001;
                xOffset = -MAP_CELL_SIZE;
            }
            else {//looking down
                rayX = (((int)playerX) / ((int)MAP_CELL_SIZE) * ((int)MAP_CELL_SIZE)) + MAP_CELL_SIZE;
                xOffset = MAP_CELL_SIZE;

            }
            rayY = (playerX - rayX) * nTan + playerY;
            yOffset = -xOffset * nTan;
        }

        //figure out when we hit a wall
        while (depth < maxDepth) {
            mapX = ((int)rayX) / ((int)MAP_CELL_SIZE);
            mapY = rayY / ((int)MAP_CELL_SIZE);
            mapIndex = mapY * MAP_ARRAY + mapX;
            if (mapIndex >= 0 && mapIndex < MAP_ARRAY * MAP_ARRAY && map[mapIndex] != 0) {//hit a wall
                depth = maxDepth;
                vWallType = map[mapIndex];
                vX = rayX;
                vY = rayY;
                vDistance = distance(playerX, vX, playerY, vY);
            }
            else {
                rayX += xOffset;
                rayY += yOffset;
                depth++;
            }
        }

        //pick the smallest of the two
        if (hDistance < vDistance) {
            rayX = hX;
            rayY = hY;
            rays[i].distance = hDistance;
            rays[i].vertical = false;
            rays[i].wallType = hWallType;
            rays[i].angle = rayAngle;
        }
        else {
            rays[i].distance = vDistance;
            rays[i].vertical = true;
            rays[i].wallType = vWallType;
            rays[i].angle = rayAngle;
        }

        //increase angle for next ray
        rayAngle += RAY_ANGLE_INC;
        if (rayAngle < 0) {//constrain angle
            rayAngle += 2 * PI;
        }
        else if (rayAngle > 2 * PI) {
            rayAngle -= 2 * PI;
        }
    }
}

//TODO collision detection
//tank controls
//not sure why we need int x and y as args
void updateMovement() {
    float nextX_w, nextY_w, nextX_s, nextY_s, nextX_a, nextY_a, nextX_d, nextY_d;
    int nextMapIndex_w, nextMapIndex_s, nextMapIndex_a, nextMapIndex_d;

    if (keyStates['w']) { // Move forward
        nextX_w = playerX + cos(playerAngle) * SPEED;
        nextY_w = playerY + sin(playerAngle) * SPEED;
        nextMapIndex_w = ((int)nextY_w / ((int)MAP_CELL_SIZE)) * MAP_ARRAY + ((int)nextX_w / ((int)MAP_CELL_SIZE));
        if (nextMapIndex_w >= 0 && nextMapIndex_w < MAP_ARRAY * MAP_ARRAY && map[nextMapIndex_w] == 0) {
            playerX = nextX_w;
            playerY = nextY_w;
        }
    }
    if (keyStates['s']) { // Move backward
        nextX_s = playerX - cos(playerAngle) * SPEED;
        nextY_s = playerY - sin(playerAngle) * SPEED;
        nextMapIndex_s = ((int)nextY_s / ((int)MAP_CELL_SIZE)) * MAP_ARRAY + ((int)nextX_s / ((int)MAP_CELL_SIZE));
        if (nextMapIndex_s >= 0 && nextMapIndex_s < MAP_ARRAY * MAP_ARRAY && map[nextMapIndex_s] == 0) {
            playerX = nextX_s;
            playerY = nextY_s;
        }
    }
    if (keyStates['a']) { // Move left
        nextX_a = playerX + sin(playerAngle) * SPEED; // Adjusting for left turn
        nextY_a = playerY - cos(playerAngle) * SPEED; // Adjusting for left turn
        nextMapIndex_a = ((int)nextY_a / MAP_CELL_SIZE) * MAP_ARRAY + ((int)nextX_a / MAP_CELL_SIZE);
        nextMapIndex_a = ((int)nextY_a / ((int)MAP_CELL_SIZE)) * MAP_ARRAY + ((int)nextX_a / ((int)MAP_CELL_SIZE));
        if (nextMapIndex_a >= 0 && nextMapIndex_a < MAP_ARRAY * MAP_ARRAY && map[nextMapIndex_a] == 0) {
            playerX = nextX_a;
            playerY = nextY_a;
        }
    }
    if (keyStates['d']) { // Move right
        nextX_d = playerX - sin(playerAngle) * SPEED; // Adjusting for right turn
        nextY_d = playerY + cos(playerAngle) * SPEED; // Adjusting for right turn
        nextMapIndex_d = ((int)nextY_d / ((int)MAP_CELL_SIZE)) * MAP_ARRAY + ((int)nextX_d / ((int)MAP_CELL_SIZE));
        if (nextMapIndex_d >= 0 && nextMapIndex_d < MAP_ARRAY * MAP_ARRAY && map[nextMapIndex_d] == 0) {
            playerX = nextX_d;
            playerY = nextY_d;
        }
    }


    // Check if player reached (14, 14) and reset maze if true

    if (playerX > MAP_CELL_SIZE * 14 && playerY > MAP_CELL_SIZE * 14) {
        score += 1;
        saveState("data/gameData.txt");
        resetMazeAndPlayer();  // Generate new maze and reset player position
    }


    glutPostRedisplay(); // Request a redraw
}

void buttons(unsigned char key, int x, int y) {
    keyStates[key] = 1; // Mark the key as pressed

    // Handle any actions that should happen only once (e.g., exit on ESC)
    if (key == 27) {  // ESC key
        if (currentState == GAME) {
            currentState = PAUSE_MENU; // Pause the game and show the menu
            glutSetCursor(GLUT_CURSOR_INHERIT);
        }
        else if (currentState == PAUSE_MENU) {
            currentState = GAME; // Resume the game
            glutSetCursor(GLUT_CURSOR_NONE);
        }
    }
}

void keyUp(unsigned char key, int x, int y) {
    keyStates[key] = 0; // Mark the key as released
}


void mouseMotion(int x, int y) {
    static int lastX = -1;
    static bool isWarping = false;

    int centerX = glutGet(GLUT_WINDOW_WIDTH) / 2;
    int centerY = glutGet(GLUT_WINDOW_HEIGHT) / 2;

    // If we're in the middle of a warp, ignore this event
    if (isWarping) {
        isWarping = false;  // Reset flag to resume regular motion
        return;
    }

    if (lastX == -1) {
        lastX = x;  // Initialize lastX to the current x on the first call
    }

    // Calculate the change in x position
    int deltaX = x - lastX;

    // Avoid excessive mouse movement in full screen, set a reasonable limit
    if (abs(deltaX) > 5) {  // You can adjust this threshold
        playerAngle += deltaX * MOUSE_SENSITIVITY;
    }

    // Warp the pointer to the center to allow for continuous movement
    isWarping = true;  // Set flag to indicate warping (avoiding looping)
    glutWarpPointer(centerX, centerY);
    lastX = centerX;  // Set lastX to center to keep deltaX calculation consistent

    glutPostRedisplay();  // Update the display
}



void menuMouse(int button, int state, int x, int y) {
    // Get the actual window size dynamically
    int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
    int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);

    // Scale mouse coordinates based on the current window size
    float scaledX = (float)x / windowWidth;
    float scaledY = (float)y / windowHeight;

    // Define button width and height dynamically based on the window size
    float buttonWidth = windowWidth * 0.2f;  // 20% of window width
    float buttonHeight = windowHeight * 0.08f;  // 8% of window height

    // Define the center positions of the buttons based on window size
    float playButtonCenterX = windowWidth / 2;
    float playButtonCenterY = windowHeight / 2 - 0.5 * buttonHeight;; // Move Play button slightly down

    // Handle Play/Resume Button
    if (scaledX > (playButtonCenterX - buttonWidth / 2) / windowWidth &&
        scaledX < (playButtonCenterX + buttonWidth / 2) / windowWidth &&
        scaledY >(playButtonCenterY - buttonHeight / 2) / windowHeight &&
        scaledY < (playButtonCenterY + buttonHeight / 2) / windowHeight) {
        // Play or Resume button clicked
        if (currentState == PAUSE_MENU) {
            currentState = GAME;  // Resume game
        }
        else {
            currentState = GAME;  // Start new game
        }
    }

    // Check for Auto-Solve button if in PAUSE_MENU
    if (currentState == PAUSE_MENU) {
        float autoSolveButtonCenterY = windowHeight / 2; // Auto-solve below Play/Resume
        if (scaledX > (playButtonCenterX - buttonWidth / 2) / windowWidth &&
            scaledX < (playButtonCenterX + buttonWidth / 2) / windowWidth &&
            scaledY >(autoSolveButtonCenterY - buttonHeight / 2) / windowHeight &&
            scaledY < (autoSolveButtonCenterY + buttonHeight / 2) / windowHeight) {
            // Auto-solve button clicked
            isTraining = true;
            // Create a POSIX thread
                pthread_t trainingThread;
                int result = pthread_create(&trainingThread, NULL, trainAgentThread, NULL);
                
                if (result == 0) {
                    pthread_detach(trainingThread);  // Detach thread to automatically clean up after completion
                } else {
                    // Handle thread creation failure if necessary
                    isTraining = false;
                    fprintf(stderr, "Failed to create training thread.\n");
                }
            currentState = GAME;
        }
    }

    // Check for Exit button (move Exit button up without overlapping Auto-Solve)
    float autoSolveButtonCenterY = windowHeight / 2;  // Auto-solve button's center
    float exitButtonCenterY = autoSolveButtonCenterY + buttonHeight; // Exit button is now above Auto-Solve
    if (scaledX > (playButtonCenterX - buttonWidth / 2) / windowWidth &&
        scaledX < (playButtonCenterX + buttonWidth / 2) / windowWidth &&
        scaledY >(exitButtonCenterY - buttonHeight / 2) / windowHeight &&
        scaledY < (exitButtonCenterY + buttonHeight / 2) / windowHeight) {
        exit(0);  // Exit button clicked
    }

    glutPostRedisplay();  // Redraw the screen after the mouse interaction
}








void menuDisplay() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Set text color to white
    glColor3f(1.0, 1.0, 1.0);

    // Display the score in the top-left corner
    std::string scoreText = "Solved Dungeons: " + std::to_string(score);
    glRasterPos2f(20, 20);
    for (const char* c = scoreText.c_str(); *c != '\0'; ++c) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
    }

    // Display "Play" or "Resume" button
    const char* buttonText = (currentState == PAUSE_MENU) ? "Resume" : "Play";
    glRasterPos2f(WINDOW_WIDTH / 2 - 20, WINDOW_HEIGHT / 2 - 8);
    for (const char* c = buttonText; *c != '\0'; ++c) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
    }

    // Conditionally display the "Auto-Solve" button only if the game is paused
    if (currentState == PAUSE_MENU) {
        glRasterPos2f(WINDOW_WIDTH / 2 - 20, WINDOW_HEIGHT / 2 + 15);
        const char* autoSolveText = "Auto-Solve";
        for (const char* c = autoSolveText; *c != '\0'; ++c) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
        }

        glRasterPos2f(WINDOW_WIDTH / 2 - 20, WINDOW_HEIGHT / 2 + 40);
        const char* exitText = "Exit";
        for (const char* c = exitText; *c != '\0'; ++c) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
        }
    }

    if (currentState != PAUSE_MENU)
    {
        // Display "Exit" button
        glRasterPos2f(WINDOW_WIDTH / 2 - 20, WINDOW_HEIGHT / 2 + 30);
        const char* exitText = "Exit";
        for (const char* c = exitText; *c != '\0'; ++c) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);
        }
    }

    glutSwapBuffers();
}



void display() {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (currentState == MENU || currentState == PAUSE_MENU) {
        menuDisplay(); // Show the main menu
        glutMouseFunc(menuMouse);
        glutSwapBuffers();
        glutSetCursor(GLUT_CURSOR_INHERIT);
        glutPassiveMotionFunc(NULL);
        glutMotionFunc(NULL);
    }
    else if (currentState == GAME) {
        // Game display code
        glutSetCursor(GLUT_CURSOR_NONE);
        glutWarpPointer(screenWidth / 2, screenHeight / 2); // Center the cursor
        glutPassiveMotionFunc(mouseMotion);
        glutMotionFunc(mouseMotion);

        castRays();
        drawView();
        drawMinimap();
    }

    glutSwapBuffers();
}


void init() {
    int actualScreenWidth = glutGet(GLUT_SCREEN_WIDTH);
    int actualScreenHeight = glutGet(GLUT_SCREEN_HEIGHT);

    // Set the dynamic screen resolution for full-screen
    screenWidth = actualScreenWidth;
    screenHeight = actualScreenHeight;
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_MULTISAMPLE);
    glEnable(GLUT_MULTISAMPLE | GL_DEPTH_TEST | GL_TEXTURE_2D | GL_BLEND);
    glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    glutCreateWindow("Dungeon Path Finder");
    glutFullScreen();
    glewInit();
    loadWallTextures();
    glClearColor(0, 0, 0, 0);
    gluOrtho2D(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0);
}

void timer(int value) {
    updateMovement();
    glutTimerFunc(16, timer, 0);  // 16 ms (60 FPS)
}


int main(int argc, char** argv) {
    srand(static_cast<unsigned>(time(0)));
    glutInit(&argc, argv);
    init();
    loadState("data/gameData.txt");
    generateMaze();
    glutDisplayFunc(display);
    glutMouseFunc(menuMouse);
    glutKeyboardFunc(buttons);
    glutKeyboardUpFunc(keyUp);  // Register key release handler
    glutTimerFunc(0, timer, 0); // Start the timer
    glutMainLoop();
    return 0;
}
