#include <pybind11/pybind11.h>
#include <optix_world.h>
#include <vector>
#include "../src/context.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rnd, module) {
    py::class_<optix::Matrix4x4>(module, "Matrix4x4", py::buffer_protocol())
        .def_buffer([](optix::Matrix4x4& m) -> py::buffer_info{
            return py::buffer_info(
                    m.getData(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    2,
                    {4, 4},
                    {sizeof(float), sizeof(float) * 4}
                );
        });

    py::class_<optix::Buffer>(module, "Buffer", py::buffer_protocol())
        .def_buffer([](optix::Buffer& m) -> py::buffer_info{
            long unsigned int width, height, depth;
            m->getSize(width, height, depth);
            py::buffer_info info(
                    m->map(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    3,
                    {width, height, depth},
                    {sizeof(float) * height, sizeof(float), sizeof(float) * width * height}
                );
            m->unmap();
            return info;
        });

    py::class_<std::vector<float, std::allocator<float> > >(module, "FloatVector", py::buffer_protocol())
        .def_buffer([](std::vector<float>& v) -> py::buffer_info{
            return py::buffer_info(
                    v.data(),
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    1,
                    {v.size()},
                    {sizeof(float)}
                );
        });

    py::class_<OptixContext, std::shared_ptr<OptixContext>>(module, "Context")
        .def(py::init<int, int>())
        .def("get_view_matrix", &OptixContext::getViewMatrix)
        .def("get_image", &OptixContext::getImage)
        .def("set_nee", &OptixContext::setNEE, py::arg("nee"))
        .def("render", &OptixContext::render, py::arg("spp"))
        .def("load_scene_file", &OptixContext::loadSceneFile, py::arg("filename"))
        .def("load_scene_json", &OptixContext::loadSceneJson, py::arg("desc"))
        .def("set_camera", &OptixContext::setCamera, py::arg("px"), py::arg("py"), py::arg("pz"), py::arg("lx"), py::arg("ly"), py::arg("lz"));
}