//Detection
#include <fstream>
#include <conio.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <windows.h>
#include <opencv2/opencv.hpp>

//WINDOW CAPTURE
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>

//BOT
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <thread>


#define KEY_9 0x39
#define KEY_8 0x38
#define KEY_7 0x37
#define KEY_6 0x36
#define KEY_5 0x35
#define KEY_4 0x34
#define KEY_3 0x33
#define KEY_2 0x32
#define KEY_1 0x31
#define KEY_0 0x30

//WINDOW CAPTURE

cv::Mat getMat(HWND hWND) {

    HDC deviceContext = GetDC(hWND);
    HDC memoryDeviceContext = CreateCompatibleDC(deviceContext);

    RECT windowRect;
    GetClientRect(hWND, &windowRect);


    //int height = windowRect.bottom; //Atuo dimentions of window/desktop
    //int width = windowRect.right;

    
    int height = 740;//Custom rectangle on screen
    int width = 1024;
    


    HBITMAP bitmap = CreateCompatibleBitmap(deviceContext, width, height);

    SelectObject(memoryDeviceContext, bitmap);

    //copy data into bitmap
    BitBlt(memoryDeviceContext, 0, 0, width, height, deviceContext, 0, 0, SRCCOPY);


    //specify format by using bitmapinfoheader!
    BITMAPINFOHEADER bi;
    bi.biSize = sizeof(BITMAPINFOHEADER);
    bi.biWidth = width;
    bi.biHeight = -height;
    bi.biPlanes = 1;
    bi.biBitCount = 32;
    bi.biCompression = BI_RGB;
    bi.biSizeImage = 0; //because no compression
    bi.biXPelsPerMeter = 1; //we
    bi.biYPelsPerMeter = 2; //we
    bi.biClrUsed = 3; //we ^^
    bi.biClrImportant = 4; //still we

    cv::Mat mat = cv::Mat(height, width, CV_8UC4); // 8 bit unsigned ints 4 Channels -> RGBA

    //transform data and store into mat.data
    GetDIBits(memoryDeviceContext, bitmap, 0, height, mat.data, (BITMAPINFO*)&bi, DIB_RGB_COLORS);

    //clean up!
    DeleteObject(bitmap);
    DeleteDC(memoryDeviceContext); //delete not release!
    ReleaseDC(hWND, deviceContext);

    return mat;
}


//OBJECT DETECTION
std::vector<std::string> load_class_list()
{
    std::vector<std::string> class_list;
    std::ifstream ifs("classes.txt"); //You need to add the path to your text file containing the name of your classes
    std::string line;
    while (getline(ifs, line))
    {
        class_list.push_back(line);
    }
    return class_list;
}


void load_net(cv::dnn::Net& net, bool is_cuda)
{
    //C:/Users/ricza/Desktop/Onnx Please/best1.onnx
    auto result = cv::dnn::readNet("best1.onnx"); //Add the path to the custom .onnx model on your own pc
    if (is_cuda)
    {
        std::cout << "Attempty to use CUDA\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else
    {
        std::cout << "Running on CPU\n";
        result.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        result.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    net = result;
}

const std::vector<cv::Scalar> colors = { cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0) };

const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float CONFIDENCE_THRESHOLD = 0.4;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void detect(cv::Mat& image, cv::dnn::Net& net, std::vector<Detection>& output, const std::vector<std::string>& className) {
    cv::Mat blob;
    int count = 0;
    auto input_image = format_yolov5(image);

    cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;

    float* data = (float*)outputs[0].data;

    const int rows = 25200 * 6;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 4; i < rows; i += 6) {
        float confidence = data[i];
        if (confidence >= CONFIDENCE_THRESHOLD) {

            float* classes_scores = data + 5;
            cv::Mat scores(1, className.size(), CV_32FC1, classes_scores);
            cv::Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            if (max_class_score > SCORE_THRESHOLD) {

                confidences.push_back(confidence);

                class_ids.push_back(class_id.x);

                float x = data[i - 4];
                float y = data[i - 3];
                float w = data[i - 2];
                float h = data[i - 1];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
            }

        }

    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i = 0; i < nms_result.size(); i++) {
        int idx = nms_result[i];
        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];
        result.box = boxes[idx];
        output.push_back(result);
    }
}


//Main Functions
//Finding nearnest object to the player
std::vector<int> nearest_object(std::vector<std::vector<int>> centers, std::vector<int> screen_center) {
    std::unordered_map<int, int> dictionary;
    for (int i = 0; i < centers.size(); i++) {
        std::vector<int> object_location = centers[i];
        //Calculate distance between an object and the character
        int distance = std::sqrt(std::pow((object_location[0] - screen_center[0]), 2) + std::pow((object_location[1] - screen_center[1]), 2));
        //Add result to dictionary
        dictionary[i] = distance;
    }

    //Sort the dictionary by distance values
    std::vector<int> sort_dictionary;
    for (const auto& elem : dictionary) {
        sort_dictionary.push_back(elem.first);
    }
    std::sort(sort_dictionary.begin(), sort_dictionary.end(), [&](int a, int b) {
        return dictionary[a] < dictionary[b];
        });

    std::vector<int> closest_object = centers[sort_dictionary[0]];
    return closest_object;
}

//Click the closest object
bool click_material(std::vector<int> closest_object, bool& is_clicking) {

    std::cout << "Moving to: " << "[" << closest_object[0] << ", " << closest_object[1] << "]" << std::endl;
    
    
    // Move the mouse cursor to the specified coordinates
    SetCursorPos(closest_object[0] , closest_object[1]);

    // Simulate a left mouse button click
    INPUT input = { 0 };
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));

    // Release the left mouse button
    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &input, sizeof(INPUT));
    
    std::this_thread::sleep_for(std::chrono::milliseconds(3500));

    std::cout << "Click!" << std::endl;

    is_clicking = false; // set the flag to false to indicate that the function has completed
    return is_clicking;
}




int main(int argc, char** argv)
{
    //Loading class names
    std::vector<std::string> class_list = {"rock"};
    
    //Loading Image
    //LPCWSTR window_title = L"Albion Online Client";
    //HWND hWND = FindWindow(NULL, window_title); //Capture Window Mode
    HWND hWND = GetDesktopWindow();  //Capture Desktop Mode

    //cv::Mat image = cv::imread("C:/Users/ricza/Desktop/Onnx Please/1.png");

    //Loading Cuda
    bool is_cuda = argc > 1 && strcmp(argv[1], "cuda") == 0;
    
    //Loading Model
    cv::dnn::Net net;
    load_net(net, is_cuda);
    std::vector<Detection> output;
    
    //Variable to set key value after keypress
    int key;
    
    time_t begin_time; //time at the start of the loop
    time_t current_time; //time when checking for a one-second interval
    double frames = 0; //number of times passed through the loop
    time(&begin_time); //sets begin_time
    
    //Define vectors for storing the closest object
    std::vector<std::vector<int>> centers;
    std::vector<int> closest_object;
    const std::vector<int> screen_center = {512, 284};

    
    //Flags to verify status of different actions
    bool is_clicking = false; //thread for moving the player
    bool display_fps = false;
    bool bot_status = false;
    bool vision_status = false;
    bool detection_status = false;

    
    while (true) {
        
        //Loading Image and processing color
        cv::Mat target = getMat(hWND);
        cv::Mat frame;
        
        cv::cvtColor(target, target, cv::COLOR_RGB2BGR);
        cv::cvtColor(target, target, cv::COLOR_BGR2RGB);
       
        target.copyTo(frame);

        //Fps Counter
        frames++; //increments frames
        time(&current_time); //sets current_time
        

        if (detection_status) {
            //Object detection
            detect(frame, net, output, class_list);
            int detections = output.size();

            //Drawing boxes and text
            for (int i = 0; i < detections; ++i) {

                //Extracting Results
                auto detection = output[i];
                auto box = detection.box;
                auto classId = detection.class_id;
                const auto color = colors[classId % colors.size()];

                //Adding the center of each object to a vector find the closes to the player
                centers.push_back({ ((box.br() + box.tl()) * 0.5).x, ((box.br() + box.tl()) * 0.5).y });


                //Rendering boxes with names
                cv::rectangle(frame, box, color, 3);
                cv::rectangle(frame, cv::Point(box.x, box.y - 20), cv::Point(box.x + box.width, box.y), color, cv::FILLED);
                cv::putText(frame, class_list[classId].c_str(), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
            }
        }
        
        
        
        //Get the nearest object to the player (relative to the center of the screen) only if we get results.
        if (centers.size() > 0 && !is_clicking && bot_status && detection_status) {
            closest_object = nearest_object(centers, screen_center);
            std::thread t(click_material, closest_object, std::ref(is_clicking));
            t.detach(); // start the thread and detach it so it runs independently
            is_clicking = true; // set the flag to true to indicate that the function is running
        }

        //Display image
        if (vision_status) {
            cv::imshow("output", frame);
            cv::waitKey(30);
        }

        
        //Empty the vectors containing the detected objects
        centers.clear();
        output.clear();
        
        //Display FPS if one second has passed since the loop started
        if (difftime(current_time, begin_time) >= 1.0){
            if (display_fps) {
                printf("Frames: %.21f\n", frames); //print the frames run through in one second
            }
            frames = 0; //reset frames
            time(&begin_time); //resets begin_time
        }


        //Event keys
        if (GetAsyncKeyState(KEY_9) & 0x8000) {
            std::cout << "Bot Activated" << std::endl;
            bot_status = true;
        }
        if (GetAsyncKeyState(KEY_8) & 0x8000) {
            std::cout << "Bot Deactivated" << std::endl;
            bot_status = false;
        }

        if (GetAsyncKeyState(KEY_7) & 0x8000) {
            std::cout << "Computer Vision On" << std::endl;
            vision_status = true;
        }
        if (GetAsyncKeyState(KEY_6) & 0x8000) {
            std::cout << "Computer Vision Off" << std::endl;
            cv::destroyAllWindows();
            vision_status = false;
        }

        if (GetAsyncKeyState(KEY_5) & 0x8000) {
            std::cout << "Detection On" << std::endl;
            detection_status = true;
        }
        if (GetAsyncKeyState(KEY_4) & 0x8000) {
            std::cout << "Detection Off" << std::endl;
            bot_status = false;
            detection_status = false;
        }

        if (GetAsyncKeyState(KEY_3) & 0x8000) {
            std::cout << "Fps On" << std::endl;
            display_fps = true;
        }
        if (GetAsyncKeyState(KEY_2) & 0x8000) {
            std::cout << "Fps Off" << std::endl;
            display_fps = false;
        }


        if (GetAsyncKeyState(KEY_0) & 0x8000) {
            std::cout << "closing..." << std::endl;
            cv::destroyAllWindows();
            break;
        }
        // add more key events here

    }
    

    return 0;
}