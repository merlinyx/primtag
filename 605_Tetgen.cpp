#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <iostream>
#include <igl/readMESH.h>
#include <igl/writeMESH.h>



using namespace std;
using namespace Eigen;
bool show_mesh = true;

// Input polygon
Eigen::MatrixXd V;
Eigen::MatrixXi F;
Eigen::MatrixXi T;
Eigen::MatrixXd B;

// Tetrahedralized interior
Eigen::MatrixXd TV;
Eigen::MatrixXi TT;
Eigen::MatrixXi TF;

// This function is called every time a keyboard button is pressed
bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier) {

  if (key >= '1' && key <= '9') {
    double t = double((key - '1')+1) / 9.0;

    VectorXd v = B.col(2).array() - B.col(2).minCoeff();
    v /= v.col(0).maxCoeff();

    vector<int> s;

    for (unsigned i=0; i<v.size();++i)
      if (v(i) < t)
        s.push_back(i);

    MatrixXd V_temp(s.size()*4,3);
    MatrixXi F_temp(s.size()*4,3);

    for (unsigned i=0; i<s.size();++i) {
      V_temp.row(i*4+0) = TV.row(TT(s[i],0));
      V_temp.row(i*4+1) = TV.row(TT(s[i],1));
      V_temp.row(i*4+2) = TV.row(TT(s[i],2));
      V_temp.row(i*4+3) = TV.row(TT(s[i],3));
      F_temp.row(i*4+0) << (i*4)+0, (i*4)+1, (i*4)+3;
      F_temp.row(i*4+1) << (i*4)+0, (i*4)+2, (i*4)+1;
      F_temp.row(i*4+2) << (i*4)+3, (i*4)+2, (i*4)+0;
      F_temp.row(i*4+3) << (i*4)+1, (i*4)+2, (i*4)+3;
    }

    viewer.data().clear();
    viewer.data().set_mesh(V_temp,F_temp);
    viewer.data().set_face_based(true);
  }

  return false;
}

int main(int argc, char *argv[])
{

  // Load a surface mesh
  igl::readMESH(TUTORIAL_SHARED_PATH "/volumetric_mesh.mesh", V, T, F);
  // cout << TUTORIAL_SHARED_PATH << endl;

  igl::opengl::glfw::Viewer viewer;
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  plugin.widgets.push_back(&menu);

  menu.callback_draw_custom_window = [&]() {
    if (ImGui::Button(show_mesh ? "Show Volumetric Mesh" : "Save Volumetric Mesh")) {
      if (show_mesh) {
        // User requests to view the tetrahedralized mesh
        igl::copyleft::tetgen::tetrahedralize(V, F, "pq1.414Y", TV, TT, TF);
        igl::barycenter(TV, TT, B);
        viewer.callback_key_down = &key_down;
        viewer.data().clear();
        viewer.data().set_mesh(TV, TF);
        viewer.data().set_face_based(true);
      } else {
        std::string filePath = "../../../volumetric_mesh.mesh";
        if (!igl::writeMESH(filePath, TV, TT, TF)) {
          std::cerr << "Failed to save the tetrahedral mesh to " << filePath << std::endl;
        } else {
          std::cout << "Tetrahedral mesh successfully saved to " << filePath << std::endl;
        }
      }
      show_mesh = !show_mesh;
    }
  };

  // Plot the generated mesh
  // viewer.callback_key_down = &key_down;
  // key_down(viewer,'5',0);
  // igl::barycenter(TV, TT, B);

  viewer.data().set_mesh(V, F);
  viewer.launch();
  return 0;
}
