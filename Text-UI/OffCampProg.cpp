#include <iostream>
#include <cstring>
#include <sstream>
#include "DomProgComm.h"
#include "react.h"
#include "OffCampProg.h"
#include "ProgArray.h"

//change
OffCampProg::OffCampProg(char* t, double c, const char* ge_init){ // note that constructor uses char* NOT bool* for GE
int i=0;
while (t[i] != 0) {                         // find the length of string t
    i++;
}
i++;                                        // i now holds the length of string t including null
title = new char[i];                        // dynamic allocation of title
for (int j = 0; j< i; j++) {                // copying each value of string t into title string
    title[j] = t[j];
}
int k = 0;
while (ge_init[k] != 0){                    // find the length of ge_init
    k++;
}
k++;                                        // k now holds the length of string ge_init including null
GE = new char[k];
for (int j = 0; j<k; j++){                  // copying each value of string ge_init into GE string
    GE[j] = ge_init[j];
}
cost = c;                                   // assign value of c to double cost
}

OffCampProg::OffCampProg(){                              // default constructor --> assign null or 0 values 
    title = new char[1];                    // dynamic allocation of title
    title[0] = 0;
    GE = new char[1];                       // dynamic allocation of GE 
    GE[0] = 0;
    cost = 0;                               // assign 0 to cost
}

OffCampProg::OffCampProg(const OffCampProg &a){          // Copy Constructor
    cost = a.cost;                          
    int j = 0;
    while (a.title[j] != 0){                // find length of a.title string
        j++;
    }
    j++;                                    // j now equals length of a.title including null
    title = new char[j];                    // dynamic allocation of new title
    for (int i = 0; i<j; i++){              // copy a.title string values into title string
        title[i] = a.title[i];
    }
    j = 0;              
    while (a.GE[j] != 0){                 // find length of a.ge_init
        j++;
    }
    j++;                                    // j now equals length of a.ge_init including null
    GE = new char[j];                       // dynamic allocation of new GE
    for (int i=0; i<j; i++){                // copy values of a.ge_init into GE
        GE[i] = a.GE[i];
    }
}

int OffCampProg::react_put_display(int loc_offset){
    int it_str_len = 0;
    int location = loc_offset;
    _put_raw(3500 + location, "Program Title: ");
    it_str_len = strlen("Program Title: ");
    location += it_str_len;
    _put_raw(3500 + location, get_title());
    it_str_len = strlen(get_title());
    location += it_str_len;
    _put_raw(3500 + location, "\n");
    location++;               // increment for the newline character

    _put_raw(3500 + location, "Program Connected GE's: ");
    it_str_len = strlen("Program Connected GE's: ");
    location += it_str_len;
    _put_raw(3500 + location, get_GE());
    it_str_len = strlen(get_GE());
    location += it_str_len;
    _put_raw(3500 + location, "\n");
    location++;               // increment for the newline character

    _put_raw(3500 + location, "Program Cost: $");
    it_str_len = strlen("Program Cost: $");
    location += it_str_len;
    stringstream example;
    example << get_cost();
    _put_raw(3500 + location, example.str().c_str());
    it_str_len = strlen(example.str().c_str());         // double is 8 bytes
    location += it_str_len;
    _put_raw(3500 + location, "\n");
    location++;               // increment for the newline character
    _put_raw(3500 + location, "\n");
    location++;               // increment for the newline character

    //ready to add more here if wanted!
    //add_yaml("OffCampProgApp1.yaml");
    return location;
  }

  int OffCampProg::react_put_display(){
    int location = 0;
    int it_str_len = 0;
    _put_raw(3500, "Program Title: ");
    it_str_len = strlen("Program Title: ");
    location += it_str_len;
    _put_raw(3500 + location, get_title());
    it_str_len = strlen(get_title());
    location += it_str_len;
    _put_raw(3500 + location, "\n");
    location++;               // increment for the newline character

    _put_raw(3500 + location, "Program Connected GE's: ");
    it_str_len = strlen("Program Connected GE's: ");
    location += it_str_len;
    _put_raw(3500 + location, get_GE());
    it_str_len = strlen(get_GE());
    location += it_str_len;
    _put_raw(3500 + location, "\n");
    location++;               // increment for the newline character

    _put_raw(3500 + location, "Program Cost: $");
    it_str_len = strlen("Program Cost: $");
    location += it_str_len;
    stringstream example;
    example << get_cost();
    _put_raw(3500 + location, example.str().c_str());
    it_str_len = strlen(example.str().c_str());         // double is 8 bytes
    location += it_str_len;
    _put_raw(3500 + location, "\n");
    location++;               // increment for the newline character
    _put_raw(3500 + location, "\n");
    location++;               // increment for the newline character

    //ready to add more here if wanted!
    //add_yaml("OffCampProgApp1.yaml");
    return location;
  }

  //void ProgArray::put_list_in_React(char* ){
   //   while()
  //}

const string OffCampProg::get_display_string_cost(){
    stringstream getting_cost;
    getting_cost << cost;
    return string(getting_cost.str());
}
const string OffCampProg::get_display_string_GE(){
    stringstream getting_GE;
    getting_GE << GE;
    return string(getting_GE.str());
}


OffCampProg& OffCampProg::operator=(const OffCampProg &b){       // Assignment Operator function
    delete [] title;        // every new must have exactly 1 delete -> object is "constructed" before assignment operator called.
    delete [] GE;           // ^ "" ditto
    cost = b.cost;
    int j = 0;
    while (b.title[j] != 0){           // finding length of "title" string
        j++;
    }
    j++;                                // j = b.title length INCLUDING NULL
    title = new char[j];
    for (int i = 0; i<j; i++){          // Dynamic allocation of new array title
        title[i] = b.title[i];
    }
    j = 0;
    while (b.GE[j] != 0){             // finding length of b.ge_init string
        j++;                            // 
    }
    j++;
    GE = new char[j];                   // NOTE: When we switch to bool* type, we already will know how long array will be 
    for (int i=0; i<j; i++){            // because there is a set "selection" of GE options
        GE[i] = b.GE[i];           // dynamically allocate and assign members of new array GE 
    }
    return *this;
}
const char* OffCampProg::get_title() const {               // returns char pointer to title, function cannot change title.
    return title;
}

const char* OffCampProg::get_GE() const {               // returns char pointer to title, function cannot change title.
    return GE;
}

const double OffCampProg::get_cost() const {               // returns double value cost, function cannot change value of cost
    //std::cerr << "Cost is " << cost << std::endl;        // testing member functions     
    return cost;
}

void OffCampProg::display_OffCampProg() const {                  // Display function will display a "program profile, listing all variables"
    std::cerr << "\n" << std::endl;
    std::cerr << "Program Name: " << title << std::endl;
    std::cerr << "Program Cost: $" << cost << std::endl;
    std::cerr << "Program GE's: " << GE << std::endl;
    //std::cerr << "\n"; << std::endl;  // for if extra spaces are needed for formatting
}

OffCampProg::~OffCampProg() {                        // deconstructor for OffCampProg class
    delete [] title;                    // dynamically de-allocates title
    delete [] GE;                       // dynamically de-allocates GE
}

void OffCampProg::display() {
    std::cerr << get_display_string();
}

void OffCampProg::put_in_global_mem(int offset) {
    _put_double(offset+4, cost);
    _put_raw(offset+12, GE);
    _put_raw(offset+20, title);
    gm_size = 20 + strlen(title)+1;
    _put_int(offset, gm_size);
}

void OffCampProg::get_from_global_mem(int offset) {
    gm_size = _get_int(offset);
    cost = _get_double(offset+4);
    GE = new char[8];
    for (int i=0; i < 8; i++) {
        GE[i] = _get_char(offset+12+i);
    }
    title = new char[gm_size - 20];
    for (int j = 0; j<gm_size-20; j++){
        title[j] = _get_char(offset+20+j);
    }
    std::cerr << strlen(GE) << std::endl; // debugging
}

char* OffCampProg::retrieve_input_string(int addstart) {
    int startingpoint = addstart;
    string ret_val_string;
    int i = 0;
    char* ret_str;
    while (_get_char(i+startingpoint) != '~'){
        ret_str[i] = _get_char(i+startingpoint);
        i++;
    }
    for (int j = startingpoint; j<i+startingpoint; j++){
        _put_char(i+startingpoint, '~');
    }
    ret_val_string = ret_str;
    return ret_str;
}

const string OffCampProg::get_display_string() {      //using stringstream object
  stringstream ss;

  ss << "Program Name: " << title << "\n" << "Program Cost: $" << cost << "\n" << "Program GE's: " << GE << "\n";
  //ss << title << " (" << GE << ")[" << cost << "]";
  return string(ss.str());
}
