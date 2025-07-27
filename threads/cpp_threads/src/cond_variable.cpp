// communication between two threads is possible through conditional variables
// sometimes we need a thread to wait till other threads are done or based on a condition
// conditional variables help with that
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;
std::mutex m;
long balance = 0;
std::condition_variable cv;

void addMoney(long money)
{
    std::lock_guard<mutex> lg(m); // acquire the mutex m
    // critical section of the code
    balance += money;
    std::cout << "balance after Money added : " << balance << std::endl;
    cv.notify_one(); // notify that we are done with the critical section and mutex is going to be released for other threads
}

void withdrawMoney(long money)
{
    // we cannot withdraw money before adding. so, there is a check if balance > 0. 
    // till then this thread waits and does not acquire the lock
    std::unique_lock<mutex> ul(m);
    cv.wait(ul, [](){return (balance != 0);});
    if (balance >= money)
    {
        balance -= money;
        std::cout << "after withdrawal, the balance is : " << balance << std::endl;
    }
    else
    {
        std::cout << "not enough money to withdraw!\n";
    }
}

int main()
{
    thread t1(withdrawMoney, 600); // function to withdraw Money from balance
    thread t2(addMoney, 500); // function to add Money to balance
    t1.join();
    t2.join();
    std::cout << "balance after the transcations is : " << balance << std::endl;
}