#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/per_face_normals.h>
#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <vector>

using namespace std;
std::ofstream outFile;

void startTracking() {
    // Open the file in overwrite mode to start fresh
    outFile.open("mouse_coordinates.txt", std::ofstream::out);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open mouse_coordinates.txt for writing." << std::endl;
    }
}

void stopTracking() {
    // Close the file when stopping tracking
    if (outFile.is_open()) {
        outFile.close();
    }
}

void writeMouseCoordsToFile(double x, double y) {
    // Write coordinates to file, assuming the file is already open
    if (outFile.is_open()) {
        outFile << x << " " << y << std::endl;
    } else {
        std::cerr << "File is not open for writing." << std::endl;
    }
}

void loadPointsFromFile(const string &filename, vector<Eigen::Vector2d> &points) {
  ifstream file(filename);
  if (!file.is_open()) {
    cerr << "Failed to open file: " << filename << endl;
    return;
  }

  string line;
  while (getline(file, line)) {
    stringstream ss(line);
    Eigen::Vector2d point;
    ss >> point[0] >> point[1];
    points.push_back(point);
  }
}

int main(int argc, char *argv[])
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd N_faces;

    // Load a mesh in OFF format
    igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off", V, F);

    igl::per_face_normals(V, F, N_faces);


    // Init the viewer
    igl::opengl::glfw::Viewer viewer;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);

    bool isDragging = false; // Flag to indicate dragging
    bool trackMouse = false; // Flag to control when to track the mouse


    // Customize the menu to include a mouse tracking toggle
    menu.callback_draw_custom_window = [&]()
    {
        if (ImGui::Button(trackMouse ? "Stop Tracking Mouse" : "Track Mouse")) {
            trackMouse = !trackMouse;
            if (trackMouse) {
              startTracking();
            } else {
              stopTracking();
            }
            isDragging = false; // Reset dragging state when toggling tracking
        }
        if (ImGui::Button("Plot Points")) {
            viewer.data().show_faces = false; // Hide the mesh faces
            // Read screen coordinates from file and plot them

            vector<Eigen::Vector2d> points;
            loadPointsFromFile("mouse_coordinates.txt", points);
            for(int i = 0; i < points.size(); i++) {
                double x = points[i](0);
                double y = points[i](1);
                Eigen::Vector3f pos(x, viewer.core().viewport(3) - y, 0);
                int fid;
                Eigen::Vector3f bc;

                if (igl::unproject_onto_mesh(pos.head<2>(), viewer.core().view, viewer.core().proj,
                                            viewer.core().viewport, V, F, fid, bc)) {
                    // Calculate the hit position using barycentric coordinates
                    Eigen::Vector3d hit_pos = V.row(F(fid, 0)).cast<double>() * bc[0] + 
                                              V.row(F(fid, 1)).cast<double>() * bc[1] + 
                                              V.row(F(fid, 2)).cast<double>() * bc[2];

                    // Adjust the hit position based on the face normal
                    Eigen::Vector3d adjusted_pos = hit_pos + N_faces.row(fid).transpose() * 0.01; // Adjust by a small value along the normal

                    // Add the adjusted point to the viewer
                    viewer.data().add_points(adjusted_pos.transpose(), Eigen::RowVector3d(1,0,0)); // Red color
                    cout << "Point " << x << " " << y << " " << endl;
                } else {
                  cout << "fail at point " << x << " " << y << " " << endl;
                }
            }
        }
    };

    // Override mouse down to start tracking if enabled
    viewer.callback_mouse_down = [&](igl::opengl::glfw::Viewer&, int, int) -> bool {
        if (trackMouse) {
            isDragging = true;
            return true;
        }
        return false;
    };

    // Override mouse move to track only when dragging
    viewer.callback_mouse_move = [&](igl::opengl::glfw::Viewer&, int x, int y) -> bool {
        if (trackMouse && isDragging) {
            writeMouseCoordsToFile(x, y);
            cout << x << " " << y << endl;
            return true;
        }
        return false;
    };

    // Override mouse up to stop tracking
    viewer.callback_mouse_up = [&](igl::opengl::glfw::Viewer&, int, int) -> bool {
        if (trackMouse) {
            isDragging = false;
            return true;
        }
        return false;
    };


    // Plot the mesh and start the viewer
    viewer.data().set_mesh(V, F);
    viewer.launch();
}
