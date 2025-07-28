// there is another method to solve race conditions for simpler problems
// atomic variables -> each operation on the variable is atomic. it prohibits operating multiple threads on atomic variable

#include <thread>
#include <iostream>
#include <mutex>
using namespace std;
// int counter = 0; race condition to access and increment this variable
atomic<int> counter = 0;

void inc_counter(int N)
{
    for(int i=0;i<N;i++)
        ++counter;
}

int main()
{
    thread t1(inc_counter, 10000);
    thread t2(inc_counter, 10000);
    t1.join();
    t2.join();
    std::cout << "counter = " << counter << endl;
}