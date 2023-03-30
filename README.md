# The-Gatherer-Cpp
This is the first version of The Gatherer ([https://github.com/Riczap/The-Gatherer](https://github.com/Riczap/The-Gatherer)) rewritten in C++ . I still have to clean the code and create header files for readability, the next step is to add an actual GUI using ImGui.

Remember to include OpenCV to the project, you can follow this tutorial if you need to: [https://www.youtube.com/watch?v=unSce_GPwto](https://www.youtube.com/watch?v=unSce_GPwto). 
**Important:** Use ***opencv_world460.lib*** for the release version.

After you have succesfully exported your custom trained model as a .onnx file you can create a .txt file containing the names of your classes as seen in the example file. I'ld advice to leave an enter space between each class name.

If you want to train and export your own custom onnx model you can follow the steps that are set up in the following Google Colab: https://colab.research.google.com/drive/19kVzBERhRwB1jywcKeJ3dALARNd5-dR7?usp=sharing


Training Data: https://drive.google.com/drive/u/2/folders/17X_f17WpzoxHMURSj5QIZ4lMUWPImf5V


Now you can *modify the path/name* of your custom files in the main.cpp file and run your inference!

You can also check the *controls.txt* file to check the temporal keybinds to activate or deactivate functions.

 - Bot Activated: 9
 - Bot Deactivated: 8
 
 - Computer Vision On: 7
 - Computer Vision Off: 6

 - Detection On: 5
 - Detection Off: 4
 
 - Fps On: 3
 - Fps Off: 2

 - Exit: 0
