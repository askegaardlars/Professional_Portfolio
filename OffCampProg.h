#ifndef _OffCampProg_H_
#define _OffCampProg_H_

#include <iostream>
using namespace std;

class OffCampProg{  // Class that represents an individual off campus study program
    //protected:
    char* title;                                // char string that holds the name of off campus program 
    char* GE;                                   // char string that holds the name of applicable GE fulfillment
    // note: Second phase will change GE to bool*, where 0 / 1 represents whether program satisfies the GE
    // each index of bool* GE will represent a different pre-defined GE (We set up key correlation)
    double cost;                                // double value that holds the numerical cost of the trip
    int gm_size;                                // length of OffCampProg stored in Global Mem -> added to support global mem functions

public:
    OffCampProg(char* t, double c, const char* ge_init);
    OffCampProg();
    OffCampProg(const OffCampProg &a);
    OffCampProg &operator=(const OffCampProg &b);
    const char* get_title() const;
    const double get_cost() const;
    const char* get_GE() const;
    void display();
    ~OffCampProg();
    void put_in_global_mem(int offset);
    void get_from_global_mem(int offset);
    const string get_display_string_cost();
    const string get_display_string_GE();
    void display_OffCampProg() const;
    const string get_display_string();        // not 100% sure that we can have const function type
    //^ used to be const string
    //const text_ui_display() const;
    int react_put_display();
    int react_put_display( int loc_offset);
    const int get_gm_size() const { return gm_size; }
    char* retrieve_input_string(int startingpoint);

    /*
    //int draw(string id, int offset);
    //char *get_ui_string(string key) { return &_global_mem[ui_params[key].as_long()]; }
    //void put_ui_string(string key, string str) { _put_raw(ui_params[key].as_long(), str.c_str()); }

    protected:
    void ui_setup(int offset);
    */

};

#endif // _OffCampProg_H_
