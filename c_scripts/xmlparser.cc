#include <cstring>
#include <iostream>
#include <memory>
#include <pugixml.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

typedef struct text_obj {
  std::string text;
  std::string tag = "";
} TextObj;

typedef struct paper {
  std::string filename = "";
  std::vector<TextObj> text_node_list;
} Paper;

std::unique_ptr<Paper> fetch_tags(std::string filepath, std::vector<std::string> tags);

PYBIND11_MODULE(xmlparser, xp) {
  py::class_<Paper>(xp, "Paper")
    .def(py::init<>())
    .def_readwrite("text_node_list", &Paper::text_node_list)
    .def_readwrite("filename", &Paper::filename);
  py::class_<TextObj>(xp, "TextObj")
    .def(py::init<>())
    .def_readwrite("text", &TextObj::text)
    .def_readwrite("tag", &TextObj::tag);
  xp.def(
    "fetch_tags",
    &fetch_tags,
    py::arg("filepath"),
    py::arg("tags"),
    py::return_value_policy::move,
    "Parse tags from xml file and return Paper object with data"
  );
}


std::string get_text(const pugi::xml_node& n) {
  std::string s;
  for (pugi::xml_node c : n.children()) {
    // if table: skip it
    if (c.type() == pugi::node_element &&
        std::strncmp(c.name(), "table-wrap", std::strlen(c.name())) == 0) {
      continue;
    }
    if (c.type() == pugi::node_pcdata || c.type() == pugi::node_cdata) {
      s += c.value(); // plain text node, add to string
    } else {
      s += get_text(c); // recursively explore child nodes
    }
  }
  return s;
}


// fetch all tags in vector tags from the file at filepath
// returns paper object which stores all requested text with assigned tag
std::unique_ptr<Paper> fetch_tags(std::string filepath, std::vector<std::string> tags) {

  // create new paper object
  auto paper = std::make_unique<Paper>();

  size_t pos = filepath.rfind('/');
  if (pos != std::string::npos) {
    paper->filename = filepath.substr(pos + 1);
  } else {
    paper->filename = filepath;
  }

  try {
    // initiate pugi stuff
    pugi::xml_document doc;
    pugi::xml_parse_result res = doc.load_file(
      filepath.c_str(),
      pugi::parse_default,
      pugi::encoding_utf8
    );
    if (!res) throw std::runtime_error(res.description());

    // loop through the tags
    for (const auto& tag : tags) {
      // add each tag to its own obj and put in paper
      std::string path = "//" + tag;
      for (auto& node : doc.select_nodes(path.c_str())) {
        TextObj t;
        t.tag = tag;
        t.text = get_text(node.node());
        paper->text_node_list.emplace_back(std::move(t));
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Skipping file due to XML parse err: " << filepath.c_str() << "\n";
    std::cerr << "Err: " << e.what() << "\n";
    return nullptr;
  }
  return paper;
}
