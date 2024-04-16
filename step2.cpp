#include <igl/readOFF.h>
#include <igl/writeOBJ.h>
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
#include <igl/bounding_box_diagonal.h>
#include <Eigen/Dense>
#include <igl/slice.h>
#include <igl/slice_into.h>


using namespace std;
std::ofstream outFile;

// Convert 2d screen coordinates into 3d points by bc
vector<Eigen::Vector3d> mapScreenPointsTo3D(const vector<Eigen::Vector2d>& screenPoints, igl::opengl::glfw::Viewer& viewer, Eigen::MatrixXd V, Eigen::MatrixXi F) {
    vector<Eigen::Vector3d> meshPoints;
    for (const auto& sp : screenPoints) {
        int fid;
        Eigen::Vector3f bc;
        Eigen::Vector3f screenPos(sp(0), viewer.core().viewport(3) - sp(1), 0.0f);
        if (igl::unproject_onto_mesh(screenPos.head<2>(), viewer.core().view, viewer.core().proj, viewer.core().viewport, viewer.data().V, viewer.data().F, fid, bc)) {
            Eigen::Vector3d p = V.row(F(fid, 0)) * bc(0) + V.row(F(fid, 1)) * bc(1) + V.row(F(fid, 2)) * bc(2);
            meshPoints.push_back(p.cast<double>());
        }
    }
    return meshPoints;
}

// functions used to calculate rod length and radius:

// double calculateLength(const vector<Eigen::Vector3d>& points) {
//     Eigen::MatrixXd pointsMatrix(points.size(), 2);
//     for (size_t i = 0; i < points.size(); ++i) {
//         pointsMatrix.row(i) = points[i];
//     }
//     // Calculate the length of the bounding box diagonal as an approximation of the rod length
//     Eigen::Vector2d min_point = pointsMatrix.colwise().minCoeff();
//     Eigen::Vector2d max_point = pointsMatrix.colwise().maxCoeff();
//     return (max_point - min_point).norm();
// }

// double calculateRodRadius(const vector<Eigen::Vector3d>& points) {
//     if (points.empty()) return 0.0;
//     Eigen::MatrixXd pointsMatrix(points.size(), 3);
//     for (size_t i = 0; i < points.size(); ++i) {
//         pointsMatrix.row(i) = points[i];
//     }
//     Eigen::Vector3d centroid = pointsMatrix.colwise().mean();
//     Eigen::MatrixXd deviations = pointsMatrix.rowwise() - centroid.transpose();
//     Eigen::Vector3d std_dev = (deviations.array().square().colwise().sum() / pointsMatrix.rows()).sqrt();
//     return std_dev.maxCoeff() / 2; // Half the max deviation as an approximation of the radius
// }

// functions used to calculate semisphere radius:

// double estimateRadiusFromPoints(const vector<Eigen::Vector3d>& points) {
//     Eigen::Vector3d minPt = Eigen::Vector3d::Constant(INFINITY);
//     Eigen::Vector3d maxPt = Eigen::Vector3d::Constant(-INFINITY);
//     for (const auto& p : points) {
//         minPt = minPt.cwiseMin(p);
//         maxPt = maxPt.cwiseMax(p);
//     }
//     double width = maxPt.x() - minPt.x(); // Horizontal extent
//     double height = maxPt.y() - minPt.y(); // Vertical extent
//     double diameter = width;
//     double radiusFromHeight = height;
//     return (diameter / 2.0 + radiusFromHeight) / 2.0; // Average the estimates
// }

// functions used to calculate width(x), length(y), height(z), center of edge:

// tuple<Eigen::Vector3d, Eigen::Vector3d> calculateCuboidParameters(const vector<Eigen::Vector3d>& points) {
//     if (points.empty()) return {Eigen::Vector3d(0, 0, 0), Eigen::Vector3d(0, 0, 0)}; // No points provided
//     Eigen::Vector3d min_corner = Eigen::Vector3d::Constant(INFINITY);
//     Eigen::Vector3d max_corner = Eigen::Vector3d::Constant(-INFINITY);
//     for (const auto& point : points) {
//         min_corner = min_corner.cwiseMin(point);
//         max_corner = max_corner.cwiseMax(point);
//     }
//     Eigen::Vector3d dimensions = max_corner - min_corner; // Width, Length, Height
//     Eigen::Vector3d center = (max_corner + min_corner) / 2; // Geometric center
//     return {dimensions, center};
// }

// functions used to calculate width(x), length(y), height(z), center of surface:
// tuple<Eigen::Vector3d, Eigen::Vector3d> calculateSurfaceParameters(const vector<Eigen::Vector3d>& points, double height) {
//     auto [dimensions, center] = calculateCuboidParameters(points);
//     dimensions(2) = 0.1;
//     return {dimensions, center};
// }


// Segmenting:
// When view is not changed, use this
void segmentRod(igl::opengl::glfw::Viewer& viewer, const vector<Eigen::Vector3d>& points) {
    if (points.empty()) {
        std::cerr << "No mesh points provided for segmentation." << std::endl;
        return;
    }

    // Get the min and max x coordinates from the collected mesh points.
    Eigen::Vector3d minPt = Eigen::Vector3d::Constant(INFINITY);
    Eigen::Vector3d maxPt = Eigen::Vector3d::Constant(-INFINITY);
    for (const auto& p : points) {
        minPt = minPt.cwiseMin(p);
        maxPt = maxPt.cwiseMax(p);
    }

    // Filter faces based on their centroid's x-coordinate.
    Eigen::MatrixXi F_new;
    std::vector<int> facesToKeep;
    for (int i = 0; i < viewer.data().F.rows(); ++i) {
        Eigen::Vector3d centroid = (
            viewer.data().V.row(viewer.data().F(i, 0)) +
            viewer.data().V.row(viewer.data().F(i, 1)) +
            viewer.data().V.row(viewer.data().F(i, 2))) / 3.0;

        // Keep faces where the centroid's x-coordinate is between minX and maxX.
        if (centroid.x() >= minPt.x() && centroid.x() <= maxPt.x() && centroid.y() >= minPt.y() && centroid.y() <= maxPt.y()) {
            facesToKeep.push_back(i);
        }
    }

    if (facesToKeep.empty()) {
        std::cerr << "No faces meet the segmentation criteria." << std::endl;
        return;
    }

    // Resize F_new to hold the faces we want to keep.
    F_new.resize(facesToKeep.size(), 3);
    for (size_t i = 0; i < facesToKeep.size(); ++i) {
        F_new.row(i) = viewer.data().F.row(facesToKeep[i]);
    }


    // Save the segmented mesh to an OBJ file.
    std::string filePath = "../../../segmented_mesh.obj";
    if (!igl::writeOBJ(filePath, viewer.data().V, F_new)) {
        std::cerr << "Failed to save the segmented mesh to " << filePath << std::endl;
    } else {
        std::cout << "Segmented mesh successfully saved to " << filePath << std::endl;
    }
}

// When view is changed, use this:
void segmentHemisphere(igl::opengl::glfw::Viewer& viewer, const vector<Eigen::Vector3d>& points) {
    if (points.empty()) {
        std::cerr << "No mesh points provided for segmentation." << std::endl;
        return;
    }

    Eigen::Vector3d minPt = Eigen::Vector3d::Constant(INFINITY);
    Eigen::Vector3d maxPt = Eigen::Vector3d::Constant(-INFINITY);
    for (const auto& p : points) {
        minPt = minPt.cwiseMin(p);
        maxPt = maxPt.cwiseMax(p);
    }

    // Filter faces based on their centroid's x-coordinate.
    Eigen::MatrixXi F_new;
    std::vector<int> facesToKeep;
    for (int i = 0; i < viewer.data().F.rows(); ++i) {
        Eigen::Vector3d centroid = (
            viewer.data().V.row(viewer.data().F(i, 0)) +
            viewer.data().V.row(viewer.data().F(i, 1)) +
            viewer.data().V.row(viewer.data().F(i, 2))) / 3.0;

        // This is the only difference compared to the above function
        if (centroid.x() >= minPt.x() && centroid.x() <= maxPt.x() && centroid.z() >= minPt.z() && centroid.z() <= maxPt.z()) {
            facesToKeep.push_back(i);
        }
    }

    if (facesToKeep.empty()) {
        std::cerr << "No faces meet the segmentation criteria." << std::endl;
        return;
    }

    // Resize F_new to hold the faces we want to keep.
    F_new.resize(facesToKeep.size(), 3);
    for (size_t i = 0; i < facesToKeep.size(); ++i) {
        F_new.row(i) = viewer.data().F.row(facesToKeep[i]);
    }


    // Save the segmented mesh to an OBJ file.
    std::string filePath = "../../../segmented_mesh.obj";
    if (!igl::writeOBJ(filePath, viewer.data().V, F_new)) {
        std::cerr << "Failed to save the segmented mesh to " << filePath << std::endl;
    } else {
        std::cout << "Segmented mesh successfully saved to " << filePath << std::endl;
    }
}



// mouse tracking functions and save:
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


// load screen points from a file writen previously
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
    
    Eigen::Vector3d segmentStart;
    Eigen::Vector3d segmentEnd;
    bool isSegmentationActive = false;


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
            // viewer.data().show_faces = false; // Hide the mesh faces
            
            // These two code update V and F of the default mesh to the uploaded mesh
            V = viewer.data().V;
            F = viewer.data().F;

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
                    // cout << "Point " << x << " " << y << " " << endl;
                    // cout << "face id " << fid << endl;
                } else {
                  cout << "fail at point " << x << " " << y << " " << endl;
                }
            }
        }
        
        if (ImGui::Button("Delete Plotted Points")) {
            viewer.data().points = Eigen::MatrixXd(0, 3);
            viewer.data().point_size = 0;  // Optionally reset the point size if needed
            viewer.data().set_points(viewer.data().points, Eigen::RowVector3d(1,0,0)); // Update point set with an empty set
        }

        if (ImGui::Button("Print Camera Matrices"))
        {
            cout << "View Matrix:\n" << viewer.core().view << "\n\n";
            cout << "Projection Matrix:\n" << viewer.core().proj << "\n";

            Eigen::Vector4f viewport = viewer.core().viewport;
            std::cout << "Viewer window size: " << viewport[2] << "x" << viewport[3] << std::endl;
        }

        if (ImGui::Button("Calculate Mesh Size")) {
            V = viewer.data().V;
            Eigen::Vector3d min_corner = V.colwise().minCoeff();
            Eigen::Vector3d max_corner = V.colwise().maxCoeff();
            
            Eigen::Vector3d size = max_corner - min_corner;

            // Print the size to the console
            std::cout << "Mesh Size: " 
                    << "width: " << size(0) 
                    << ", height: " << size(1) 
                    << ", depth: " << size(2) 
                    << std::endl;
        }
        
        if (ImGui::Button("Color Hit Faces")) {
            vector<Eigen::Vector2d> points;
            loadPointsFromFile("mouse_coordinates.txt", points);
            
            // Prepare to color hit faces
            Eigen::MatrixXd C(viewer.data().F.rows(), 3); // Create a color matrix for faces
            C.setConstant(viewer.data().F.rows(), 3, 0.8); // Default color for all faces to light grey
            for (int i = 0; i < points.size(); i++) {
                double x = points[i](0) ;
                double y = points[i](1) ;
                Eigen::Vector3f pos(x, viewer.core().viewport(3) - y, 0);
                int fid;
                Eigen::Vector3f bc;
                if (igl::unproject_onto_mesh(pos.head<2>(), viewer.core().view, viewer.core().proj,
                                            viewer.core().viewport, viewer.data().V, viewer.data().F, fid, bc)) {
                    C.row(fid) = Eigen::RowVector3d(1, 0, 0); // Red color for hit faces
                    std::cout << "Hit at face id: " << fid << std::endl;
                }
            }
            viewer.data().set_colors(C);
        }

        // if (ImGui::Button("Calculate Rod Parameters")) {
        //     vector<Eigen::Vector2d> points;
        //     loadPointsFromFile("mouse_coordinates.txt", points);
        //     auto meshPoints = mapScreenPointsTo3D(points, viewer, viewer.data().V, viewer.data().F);
        //     double length = calculateLength(meshPoints); // Wrong, can't directly use screen coordinates
        //     double radius = calculateRodRadius(meshPoints); // check to see whether this time it works, already make, not run yet
        //     cout << "Rod Length: " << length << " units" << endl;
        //     cout << "Average Radius: " << radius << " units" << endl;
        // }

        // if (ImGui::Button("Calculate Hemisphere Radius")) {
        //     vector<Eigen::Vector2d> points;
        //     loadPointsFromFile("mouse_coordinates.txt", points);
        //     auto meshPoints = mapScreenPointsTo3D(points, viewer, viewer.data().V, viewer.data().F);
        //     double radius = estimateRadiusFromPoints(meshPoints);
        //     cout << "Estimated Radius of Hemisphere: " << radius << " units" << endl;
        // }

        // if (ImGui::Button("Calculate Edge Parameters")) {
        //     vector<Eigen::Vector2d> points;
        //     loadPointsFromFile("mouse_coordinates.txt", points);
        //     auto meshPoints = mapScreenPointsTo3D(points, viewer, viewer.data().V, viewer.data().F);
        //     auto [dimensions, center] = calculateCuboidParameters(meshPoints);
        //     cout << "Dimensions (Width, Length, Height): " << dimensions.transpose() << endl;
        //     cout << "Center: " << center.transpose() << endl;
        // }

        // if (ImGui::Button("Calculate Surface Parameters")) {  // not checked yet.
        //     vector<Eigen::Vector2d> points;
        //     loadPointsFromFile("mouse_coordinates.txt", points);
        //     auto meshPoints = mapScreenPointsTo3D(points, viewer, viewer.data().V, viewer.data().F);
        //     auto [dimensions, center] = calculateSurfaceParameters(meshPoints, 0.1);
        //     cout << "Dimensions (Width, Length, Height): " << dimensions.transpose() << endl;
        //     cout << "Center: " << center.transpose() << endl;
        // }

        if (ImGui::Button("DraggedView Segmentation")) {
            // Load screen points from file
            vector<Eigen::Vector2d> points;
            loadPointsFromFile("mouse_coordinates.txt", points);
            auto meshPoints = mapScreenPointsTo3D(points, viewer, viewer.data().V, viewer.data().F);
            segmentRod(viewer, meshPoints);
        }
        if (ImGui::Button("Non-DraggedView Segmentation")) {
            // Load screen points from file
            vector<Eigen::Vector2d> points;
            loadPointsFromFile("mouse_coordinates.txt", points);
            auto meshPoints = mapScreenPointsTo3D(points, viewer, viewer.data().V, viewer.data().F);
            segmentHemisphere(viewer, meshPoints);
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