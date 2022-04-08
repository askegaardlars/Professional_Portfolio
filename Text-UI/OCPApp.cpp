#include"../react.h"
#include <cstring>
#include <sstream>
#include "OffCampProg.h"
#include "OffCampProg.cpp"



int main() {
  _init();

  int location = 0;
  int it_str_len = 0;
  OffCampProg new_trip1{"Biology and Environmental Sustainability in Guatamala", 12832, "WRI"};
  OffCampProg new_trip4{"Government and Public Health in Ecuador", 193834, "MCG, ALS-A"};
  OffCampProg new_trip6{"Economics and Governmental systems in the Czech Republic", 23429, "HBS"};
  OffCampProg new_trip7{"International Language Studies in Context: Madrid, Spain", 19832, "IST or HBS"};
  OffCampProg new_trip8{"Immersive experience in Culture and Religion in Jordan", 9283, "BTS-B and BTS-T"};
  _put_raw(3000 + 50, "Select a program to view\n");
  _put_raw(3000 + 100, "Guatamala\n\0");
  _put_raw(3000 + 150, "Ecuador\n\0");
  _put_raw(3000 + 200, "Czech Republic\n\0");
  _put_raw(3000 + 250, "Spain");
  _put_raw(3000 + 300, "Jordan");

  _put_char(3001, '1');
  new_trip6.react_put_display();

  if (_get_char(3000 + 1) == '~'){
    add_yaml("OffCampProgApp2.yaml");
  }
  if (_get_char(3000 + 1) == '1') {
    new_trip6.react_put_display();
  }
 
  if (_get_char(3000 + 1) == '2') {
    new_trip6.react_put_display();
  }

  if (_get_char(3000 + 1) == '3') {
    new_trip6.react_put_display();
  }
  
  if (_get_char(3000 + 1) == '4') {
    new_trip7.react_put_display();
  }

  if (_get_char(3000 + 1) == '5') {
    new_trip8.react_put_display();
  }

  if (_received_event()) {
    if (_event_id_is("Program_picker"){
      _put_char(3000 + 1, '~');
    }
  }

  _quit();
  
}
