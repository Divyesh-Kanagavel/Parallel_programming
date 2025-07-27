// semaphore is a signalling mechanism through which threads communicate and synchronize
// in this example,we see binary_semaphore

#include <iostream>
#include <thread>
#include <semaphore>
#include <chrono>

using namespace std;

std::binary_semaphore smph_MainToThread {0}; // both semaphore are locked
std::binary_semaphore smph_ThreadToMain {0};

void func()
{
    smph_MainToThread.acquire(); // tries to acquire the semaphore
    std::cout << "inside func() \n";
    this_thread::sleep_for(chrono::seconds(2));
    smph_ThreadToMain.release(); // release the ThreadToMain semaphore for use by other threads

}

int main()
{
    std::cout << "inside main() \n";
    thread t(func); // thread created but blocked since MainToThread is still locked.
    std::cout << "send signals from main to t\n";
    smph_MainToThread.release(); // this semaphore is released, so the func in thread t can now acqurie this semaphore
    cout << "in main, trying to acquire ThreadToMain \n";
    smph_ThreadToMain.acquire();
    std::cout << "got the signal \n";
    t.join();
    return 0;

}