// for functions with recursion, there is a separate class of mutex called the
// recursion mutex. which creates locks for each call of recursion and unlocks as stack unwinds

#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>

using namespace std;

int buffer = 0;
std::recursive_mutex m;

void recursion(char c, int loopFor)
{
    if (loopFor < 0)
        return;
    m.lock();
    cout << "thread id = " << c << " " << buffer++ << endl;
    recursion(c, --loopFor);
    m.unlock();
    cout << "unlocked by thread ID : " << c << endl;
}

int main()
{
    thread t1(recursion, '1', 2);
    thread t2(recursion, '2', 2);
    t1.join();
    t2.join();
    return 0;
}