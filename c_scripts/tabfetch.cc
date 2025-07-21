#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <dirent.h>
#include <iostream>
#include <pugixml.hpp>
#include <pybind11/pybind11.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>

namespace py = pybind11;

typedef struct xml_node {
  struct xml_node *next = NULL;
  struct table_node *tab_head = NULL;
  std::string pmcid = "";
} Xml;

typedef struct table_node {
  std::string text = "";
  struct xml_node *parent = NULL;
  struct table_node *next = NULL;
} Table;

Table * parse_text(std::string filepath, Xml *parent);
Xml * read_file(const char * subdir);

// pybind module
PYBIND11_MODULE(tabfetch, tf) {
  // structs -> classes
  py::class_<Xml>(tf, "Xml")
    .def(py::init<>())
    .def_readwrite("next", &Xml::next)
    .def_readwrite("tab_head", &Xml::tab_head)
    .def_readwrite("pmcid", &Xml::pmcid);

  py::class_<Table>(tf, "Table")
    .def(py::init<>())
    .def_readwrite("text", &Table::text)
    .def_readwrite("parent", &Table::parent)
    .def_readwrite("next", &Table::next);

  // functions
  tf.def(
    "parse_text",
    &parse_text,
    py::arg("filepath"),
    py::arg("parent"),
    py::return_value_policy::take_ownership,
    "Parse <table-wrap> from XML file and return head of Table* linked list"
  );

  tf.def(
    "read_file",
    &read_file,
    py::arg("subdir"),
    py::return_value_policy::take_ownership,
    "Read all XMLs in fulltexts/<subdir> and return head of Xml* circular linked list"
  );

}

// function to parse tables from xml files
Table * parse_text(std::string filepath, Xml *parent) {

  // head node of table linked list
  Table *head = NULL;

  pugi::xml_document doc;
  pugi::xml_parse_result res = doc.load_file(
    filepath.c_str(),
    pugi::parse_default,
    pugi::encoding_utf8
  );
  if (!res) throw std::runtime_error(res.description());

  for (auto& node : doc.select_nodes("//table-wrap")) {
    Table *new_tab = new Table;
    std::ostringstream oss;
    node.node().print(
      oss,
      "",
      pugi::format_default,
      pugi::encoding_utf8
    );
    new_tab->text = oss.str();
    new_tab->parent = parent;

    if (head == NULL) {
      head = new_tab;
      continue;
    } else {
      Table *cur = head;
      while (cur->next != NULL) {
        cur = cur->next;
      }
      cur->next = new_tab;
    }
  }

  return head;

}

// function to read each xml file and get their tables
Xml * read_file(const char * subdir) {

  // head node of Xml circular linked list
  Xml *head = NULL;

  DIR *dir = NULL;
  struct dirent *ent;
  std::string dirname = std::string(subdir);
  if ((dir = opendir(dirname.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {

      // skip hidden files ("." and ".." usually)
      if (ent->d_name[0] == '.') continue;

      Xml *new_xml = new Xml;

      // get any tables from the text
      std::string filename = ent->d_name;
      std::string filepath = dirname + "/" + filename;

      try {
        new_xml->tab_head = parse_text(filepath, new_xml);
      } catch (const std::exception &e) {
        std::cerr << "Skipping file due to XML parse err: " << filepath << "\n";
        std::cerr << "Err: " << e.what() << "\n";
        delete new_xml;
        continue;
      }

      // get the pmcid (file name)
      size_t pos = std::string(filename).find('.');
      std::string pmcid = std::string(filename).substr(0, pos);
      new_xml->pmcid = pmcid;


      // add to list
      if (head == NULL) {
        head = new_xml;
        new_xml->next = head;
      } else {
        Xml *cur = head;
        while (cur->next != head) {
          cur = cur->next;
        }
        cur->next = new_xml;
        new_xml->next = head;
      }
    }
  }
  closedir(dir);
  return head;
}

