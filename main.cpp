#include <igl/opengl/glfw/Viewer.h>
#include <igl/copyleft/tetgen/tetrahedralize.h>
#include <igl/readOFF.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>

using namespace std;
using namespace Eigen;

// Input polygon
Eigen::MatrixXd V;
Eigen::MatrixXi F;
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
  using namespace Eigen;
  using namespace std;

  // Load a surface mesh
  igl::readOFF(TUTORIAL_SHARED_PATH "/bunny.off",V,F);

  // Init the viewer
  igl::opengl::glfw::Viewer viewer;

  // Attach a menu plugin
  igl::opengl::glfw::imgui::ImGuiPlugin plugin;
  viewer.plugins.push_back(&plugin);
  igl::opengl::glfw::imgui::ImGuiMenu menu;
  plugin.widgets.push_back(&menu);
  menu.callback_draw_custom_window = [&]() {
    V = viewer.data().V;
    F = viewer.data().F;
    if (ImGui::Button("Save tetrahedralized mesh")) {
      Eigen::MatrixXd TV; // Output vertices
      Eigen::MatrixXi TT; // Output tetrahedral indices
      Eigen::MatrixXi TF; // Output triangle faces
      string switches = "pq1.414a0.01";
      V = viewer.data().V;
      F = viewer.data().F;
      int numRegions;

      if (igl::copyleft::tetgen::tetrahedralize(V, F, switches, TV, TT, TF) != 0) {
          std::cerr << "Tetrahedralization failed." << std::endl;
          return;
      }
      std::string filePath = "../../../volumetric_mesh.obj";
      if (!igl::writeOBJ(filePath, TV, TF)) {
          std::cerr << "Failed to save the segmented mesh to " << filePath << std::endl;
      } else {
          std::cout << "Segmented mesh successfully saved to " << filePath << std::endl;
      }
    }
  };
  // Tetrahedralize the interior
  igl::copyleft::tetgen::tetrahedralize(V,F,"pq1.414Y", TV,TT,TF);

  // Compute barycenters
  igl::barycenter(TV,TT,B);

  // Plot the generated mesh
  viewer.callback_key_down = &key_down;
  key_down(viewer,'5',0);
  viewer.launch();
}