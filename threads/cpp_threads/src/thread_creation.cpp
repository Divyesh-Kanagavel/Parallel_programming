// different types of thread creation mechanisms.
// in odd_even_sum_threded.cpp, we used function pointers
// here, we explore other alternate methods

#include <iostream>
#include <thread>
using namespace std;

// functor object passed to std::thread
class Base
{
    public:
        void operator ()(int x)
        {
            std::cout << "inside functor !\n";
            while (x-- > 0)
                std::cout << "x = " << x << std::endl;
        }

};

// non static member function
class NStatic
{
    public:
        void run(int x)
        {
            std::cout << "inside non static !\n";
            while (x -- > 0)
                std::cout << "x = " << x << std::endl;
        }
};

// Static member function
class Static
{
    public:
    static void run(int x)
    {
        std::cout << "inside static !\n";
        while (x -- > 0)
            std::cout << "x = " << x << std::endl;
    }

};
int main()
{
    NStatic ns;
    // lambda function passed to std::thread
    thread t([](int x){
       std::cout << "inside lambda \n";
       while(x-- > 0)
           cout << "x = " << x << std::endl;
    }, 10);
    thread t2(Base(), 10);
    thread t3(&NStatic::run, &ns, 10);
    thread t4(&Static::run, 10);
    t.join();
    t2.join();
    t3.join();
    t4.join();


}