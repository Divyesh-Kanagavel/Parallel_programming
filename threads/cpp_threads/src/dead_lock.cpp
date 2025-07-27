// illustration of dead locks and how it can be solved using std::lock
#include <iostream>
#include <thread>
#include <mutex>
using namespace std;
mutex m1, m2;
void task_a()
{
   // m1.lock(); // possibility of deadlock, to solve either lock in the same order in both threads, or use std::lock
    // std::lock takes in multiple mutexes to lock, if any of the mutexes is already locked, it takes out other mutexes and waits till the locked mutex is unlocked. this way there is no deadlock
    //m2.lock();
    std::lock(m1, m2);
    this_thread::sleep_for(chrono::seconds(2));
    for(int i=0;i<1000;i++)
    std::cout << "task_a!\n";
    m1.unlock();
    m2.unlock();
}

void task_b()
{
   // m2.lock(); 
   // m1.lock();
   std::lock(m2, m1);
    this_thread::sleep_for(chrono::seconds(2));
    for(int i=0;i<1000;i++)
    std::cout << "task_b!\n";
    m2.unlock();
    m1.unlock();
}

int main()
{
    thread t1(task_a);
    thread t2(task_b);
    t1.join();
    t2.join();
    return 0;
}