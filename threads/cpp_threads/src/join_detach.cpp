/*
join() and detach() illustration
always check with joinable() before calling join() or detach()
because double join() or detach() will crash
*/

#include <iostream>
#include <thread>
#include <chrono>
using namespace std;

void print(int x)
{
    while (x-- > 0)
    {
        cout << "x = " << x << endl;
    }
}


int main()
{
    cout << "main() function!\n";
    thread t1(print, 10);
    if (t1.joinable())
        //t1.join(); // print is completed before next cout statement
        t1.detach(); // thread detached from main , so main's print statements could be printed before thread print is executed.
    cout << "main after thread join!\n";
    this_thread::sleep_for(chrono::seconds(3));

    
}