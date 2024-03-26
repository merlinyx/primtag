// Used the tutorial 106_ViewerMenu 
#include <igl/readOFF.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;


igl::opengl::glfw::Viewer viewer;
bool is_mouse_down = false;
Eigen::MatrixXd V, C;
Eigen::MatrixXi F;
std::ofstream myfile;

int main(int argc, char *argv[])
{
  
  bool trackMouse = false;

  // Load a mesh in OFF format
  igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off", V, F);

  Eigen::Vector3d m = V.colwise().minCoeff();
  Eigen::Vector3d M = V.colwise().maxCoeff();

  // Corners of the bounding box
  Eigen::MatrixXd V_box(2,2);
  V_box <<
  0, -1, //viewer.current_mouse_x, viewer.current_mouse_y, //m(0), m(1),
  -1, 0; //viewer.current_mouse_y, viewer.current_mouse_x; //M(0), m(1);

  // Edges of the bounding box
  Eigen::MatrixXi E_box(1,2);
  E_box <<
  0, 1;

  // Attach a menu plugin
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  plugin.widgets.push_back(&menu);

  // Set mouse down callback
    viewer.callback_mouse_down = [&trackMouse](igl::opengl::glfw::Viewer&, int, int)->bool {
        is_mouse_down = true; // Mouse is being held down
        if (trackMouse == true) {
          Eigen::Vector3d m = V.colwise().minCoeff();
          return true;
        }
        return false;
    };

    // Set mouse up callback
    viewer.callback_mouse_up = [](igl::opengl::glfw::Viewer&, int, int)->bool {
        is_mouse_down = false; // Mouse is no longer being held down
        return false; // Return false to allow further processing
    };

  // Customize the menu
  double doubleVariable = 0.1f; // Shared between two menus

  // My dropdown
  menu.callback_draw_viewer_menu = [&]()
  {
    // Expose an enumeration type
    enum Primitive { Choose=0, Hook, Hole, Rod, Tube, Hemisphere, Edge, Clip, Surface };
    static Primitive primative = Choose;
    ImGui::Combo("Primitive", (int *)(&primative), "Choose\0Hook\0Hole\0Rod\0Tube\0Hemisphere\0Edge\0Clip\0Surface\0\0");

    // Add a button
    if (ImGui::Button("Print Camera Matrix", ImVec2(-2, 0)))
    {
      std::cout << viewer.core().view;
    }

    // Add a button
    if (ImGui::Button("Print Primative Type", ImVec2(-1,0)))
    {
      std::cout << primative;
      std::cout << "\n";
    }


    // Add a button
    if (ImGui::Button("Track Mouse coords", ImVec2(-1,0)))
    {
      trackMouse = !trackMouse;
      if (trackMouse) {
        myfile.open("partOne.txt"); //NAME OF THE FILE
        myfile << viewer.core().view;
        myfile << "\n";
        myfile << primative;
        myfile << "\n";
        std::cout << viewer.core().view;
        std::cout << "\n";
        std::cout << primative;
        std::cout << "\n";
      } else {
        myfile.close();
      }
    }

    if (trackMouse && is_mouse_down) {
      myfile << viewer.current_mouse_x;
      myfile << ", ";
      myfile << viewer.current_mouse_y;
      myfile << "\n";
      std::cout << viewer.current_mouse_x;
      std::cout << ", ";
      std::cout << viewer.current_mouse_y;
      std::cout << "\n";
    }
  };

  // Plot the mesh
  viewer.data().set_mesh(V, F);
  viewer.data().add_label(viewer.data().V.row(0) + viewer.data().V_normals.row(0).normalized()*0.005, "Hello World!");

  // Plot the corners of the bounding box as points
  viewer.data().add_points(V_box,Eigen::RowVector3d(1,0,0));

  // Plot the edges of the bounding box
  for (unsigned i=0;i<E_box.rows(); ++i)
    viewer.data().add_edges
    (
      V_box.row(E_box(i,0)),
      V_box.row(E_box(i,1)),
      Eigen::RowVector3d(1,0,0)
    );

  viewer.launch();
}
